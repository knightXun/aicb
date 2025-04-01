import yaml
import json
import sys
from utils import print_memory_breakdown

def load_config_from_content(content):
    try:
        # Try parsing as JSON first
        try:
            config = json.loads(content)
            # Check if this is a multimodal model with text_config
            if 'text_config' in config:
                # Use text_config for model parameters
                text_config = config['text_config']
                return {
                    'hidden_size': text_config['hidden_size'],
                    'num_layers': text_config['num_hidden_layers'],
                    'vocab_size': config.get('vocab_size', 256000),  # Default for multimodal models
                    'intermediate_size': text_config['intermediate_size'],
                    'seq_len': 2048,  # Default value since not in config
                    'mbs': 1,        # Default value
                    'batch_accum': 1, # Default value
                    'tp': 1,         # Default value
                    'pp': 1,         # Default value
                    'dp': 1,         # Default value
                    'zero_stage': 0,  # Default value
                    'tie_word_embeddings': config.get('tie_word_embeddings', True),
                    'num_attention_heads': text_config['num_attention_heads'],
                    'num_key_value_heads': text_config.get('num_key_value_heads', text_config['num_attention_heads']),
                    'full_checkpointing': False  # Default value
                }
            else:
                # Original code for non-multimodal models
                return {
                    'hidden_size': config['hidden_size'],
                    'num_layers': config['num_hidden_layers'],
                    'vocab_size': config['vocab_size'],
                    'intermediate_size': config['intermediate_size'],
                    'seq_len': 2048,  # Default value since not in config
                    'mbs': 1,        # Default value
                    'batch_accum': 1, # Default value
                    'tp': 1,         # Default value
                    'pp': 1,         # Default value
                    'dp': 1,         # Default value
                    'zero_stage': 0,  # Default value
                    'tie_word_embeddings': config.get('tie_word_embeddings', True),
                    'num_attention_heads': config['num_attention_heads'],
                    'num_key_value_heads': config.get('num_key_value_heads', config['num_attention_heads']),
                    'full_checkpointing': False  # Default value
                }
        except json.JSONDecodeError:
            # If not JSON, try YAML
            config = yaml.safe_load(content)
            
            # Extract relevant parameters from YAML config
            model_config = config['model']['model_config']
            parallelism = config['parallelism']
            tokens = config['tokens']
            optimizer = config['optimizer']
            
            return {
                'hidden_size': model_config['hidden_size'],
                'num_layers': model_config['num_hidden_layers'],
                'vocab_size': model_config['vocab_size'],
                'intermediate_size': model_config['intermediate_size'],
                'seq_len': tokens['sequence_length'],
                'mbs': tokens['micro_batch_size'],
                'batch_accum': tokens['batch_accumulation_per_replica'],
                'tp': parallelism['tp'],
                'pp': parallelism['pp'],
                'dp': parallelism['dp'],
                'zero_stage': optimizer['zero_stage'],
                'tie_word_embeddings': model_config['tie_word_embeddings'],
                'num_attention_heads': model_config['num_attention_heads'],
                'num_key_value_heads': model_config.get('num_key_value_heads', model_config['num_attention_heads']),
                'full_checkpointing': optimizer.get('full_checkpointing', False)  # Renamed from fsdp_checkpointing
            }
    except Exception as e:
        print(f"Error parsing configuration: {str(e)}")
        return None

def load_config_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return load_config_from_content(content)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <config_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    config = load_config_from_file(file_path)

    if config:
        print_memory_breakdown(**config)

if __name__ == "__main__":
    main()
