import numpy as np
import functools

@functools.lru_cache(maxsize=None)
def get_num_hidden_layers_in_pp(hidden_size, num_layers, vocab_size, intermediate_size, num_attention_heads, pp_size):
    if pp_size == 1:
        return num_layers

    # Get list of pipeline blocks and their costs
    pipeline_blocks = []
    block_costs = []
    
    # Embedding layer (treated as zero cost in the original implementation)
    pipeline_blocks.append("embedding")
    block_costs.append(0)
    
    # Decoder layers
    decoder_cost = (4 * num_attention_heads * (hidden_size//num_attention_heads) * hidden_size + 
                   3 * intermediate_size * hidden_size)
    for _ in range(num_layers):
        pipeline_blocks.append("decoder")
        block_costs.append(decoder_cost)
    
    # LM head
    pipeline_blocks.append("lm_head")
    block_costs.append(vocab_size * hidden_size)
    
    # Now follow the same logic as the original code
    total_cost = sum(block_costs)
    target_cost_per_rank = total_cost / pp_size
    
    blocks_in_rank0 = 0
    current_cost = 0
    
    for block_idx, block_cost in enumerate(block_costs):
        current_cost += block_cost
        blocks_in_rank0 += 1
        
        # Check if we should move to next rank
        remaining_ranks = pp_size - 1  # -1 because we're calculating for rank 0
        remaining_nonzero_blocks = sum(1 for c in block_costs[block_idx+1:] if c > 0)
        
        if (remaining_ranks > 0 and remaining_nonzero_blocks <= remaining_ranks) or (current_cost >= target_cost_per_rank):
            break
            
    num_hidden_layers_in_pp = blocks_in_rank0 - 1 # We exclude first rank as it's the embedding layer
    return num_hidden_layers_in_pp

@functools.lru_cache(maxsize=None)
def calculate_memory_components(
    hidden_size, num_attention_heads, num_key_value_heads, num_layers, vocab_size, intermediate_size,
    seq_len, mbs, batch_accum, tp, pp, dp, zero_stage,
    tie_word_embeddings, full_checkpointing=False
):
    # Calculate base components first
    if pp == 1:
        num_hidden_layers_in_pp = num_layers
    else:
        num_hidden_layers_in_pp = get_num_hidden_layers_in_pp(hidden_size, num_layers, vocab_size, intermediate_size, num_attention_heads, pp)
    
    # Model BF16 calculation
    vocab_embeddings = vocab_size * hidden_size * (2 if (not tie_word_embeddings and pp==1) else 1)
    
    layer_params = (
        (hidden_size * hidden_size * (1 + 2*num_key_value_heads/num_attention_heads))  # qkv_proj
        + (hidden_size * hidden_size)     # out_proj
        + (hidden_size * 2 * intermediate_size)  # gate_up_proj
        + (intermediate_size * hidden_size)      # down_proj
    )
    
    model_bf16_full = (vocab_embeddings + num_hidden_layers_in_pp * layer_params) * (2 / 1024 / 1024) / tp

    # Calculate number of parameters in billions
    num_params_in_B = (vocab_embeddings + num_layers*layer_params) / 1e9

    # Adjust model components based on ZeRO stage
    if zero_stage == 3:
        # In ZeRO-3, model parameters are sharded across dp ranks
        model_bf16 = model_bf16_full / dp
        fp32_params = 2 * model_bf16
        fp32_grads = 2 * model_bf16
        optimstates = 4 * model_bf16
        # Additional communication buffers for ZeRO-3
        zero3_buffers = 2 * model_bf16  # For parameter gathering during forward/backward
    else:
        # For ZeRO-0/1/2
        dp_if_zero = 1 if zero_stage == 0 else dp
        model_bf16 = model_bf16_full
        fp32_params = 2 * model_bf16 / dp_if_zero
        fp32_grads = 2 * model_bf16
        optimstates = 4 * model_bf16 / dp_if_zero
        zero3_buffers = 0

    use_ddp = zero_stage == 0 and dp > 1
    ddp_grads_buffers = model_bf16 if use_ddp else 0
    overhead = 72 + 32 * mbs

    # Activations calculation with FSDP checkpointing support
    is_mha = num_key_value_heads == num_attention_heads
    decoder_layer_mib = (seq_len * mbs * hidden_size/tp) * (2/1024/1024) * (4*intermediate_size/hidden_size + 6 + 2*num_key_value_heads/num_attention_heads + 2)
    
    if pp > 1:
        activs = min(pp, batch_accum) * num_hidden_layers_in_pp * decoder_layer_mib
    else:
        cast_to_fp32 = sharded_cross_entropy = seq_len * mbs * vocab_size * (2 / 1024 / 1024) * 2 / tp
        base_activs = num_layers * decoder_layer_mib + cast_to_fp32 + sharded_cross_entropy
        
        # Apply activation reduction for FSDP checkpointing in ZeRO-3
        if zero_stage == 3 and full_checkpointing:
            activs = base_activs / dp  # Activation memory is reduced by dp factor with checkpointing
        else:
            activs = base_activs

    # Calculate aggregate metrics
    memory_usage_after_optimstates = (
        model_bf16 + 
        fp32_params + 
        fp32_grads + 
        optimstates + 
        ddp_grads_buffers + 
        zero3_buffers +
        overhead
    )

    memory_usage_before_optimstates = (
        model_bf16 + 
        fp32_params + 
        fp32_grads + 
        ddp_grads_buffers +
        zero3_buffers
    )

    memory_usage_peak_tbi = (
        model_bf16 + 
        fp32_params + 
        fp32_grads + 
        optimstates + 
        ddp_grads_buffers + 
        zero3_buffers +
        overhead + 
        activs
    )

    return {
        "Components": {
            "Model BF16": model_bf16,
            "FP32 Parameters": fp32_params,
            "FP32 Gradients": fp32_grads,
            "Optimizer States": optimstates,
            "DDP Gradient Buffers": ddp_grads_buffers,
            "ZeRO-3 Buffers": zero3_buffers,
            "Overhead": overhead,
            "Activations": activs,
        },
        "Aggregates": {
            "Memory Before Optimizer States": memory_usage_before_optimstates,
            "Memory After Optimizer States": memory_usage_after_optimstates,
            "Peak Memory (TBI)": memory_usage_peak_tbi
        }
    }

def print_memory_breakdown(
    hidden_size, num_attention_heads, num_key_value_heads, num_layers, vocab_size, intermediate_size,
    seq_len, mbs, batch_accum, tp, pp, dp, zero_stage,
    tie_word_embeddings, full_checkpointing=False
):
    results = calculate_memory_components(
        hidden_size, num_attention_heads, num_key_value_heads, num_layers, vocab_size, intermediate_size,
        seq_len, mbs, batch_accum, tp, pp, dp, zero_stage,
        tie_word_embeddings, full_checkpointing
    )
    memory_usage_peak_tbi = results["Aggregates"]["Peak Memory (TBI)"]
    
    # Print components
    print("Memory Component Breakdown:")
    components = results["Components"]
    for name, value in components.items():
        print(f"{name}: {value:.1f} MiB")
    
    # Print aggregates
    print("\nAggregate Memory Metrics:")
    aggregates = results["Aggregates"]
    for name, value in aggregates.items():
        print(f"{name}: {value:.1f} MiB")
    
    # TODO: OOM prediction
    # oom_prediction = "OOM" if memory_usage_peak_tbi > 75000 else "No OOM"
    # print(f"\nOOM Prediction: {oom_prediction}")
