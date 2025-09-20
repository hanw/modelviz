#!/usr/bin/env python3
"""
Generic Model Weight Visualization Library

A reusable library for visualizing weights and attention patterns of transformer models.
Supports any Hugging Face model with comprehensive visualization capabilities.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class ModelVisualizer:
    """Generic weight visualization for any Hugging Face transformer model."""
    
    def __init__(self, model_id: str, output_dir: str = "./model_visualizations", 
                 model_type: str = "auto", trust_remote_code: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            model_id: Hugging Face model ID
            output_dir: Directory to save visualization outputs
            model_type: Type of model (auto, protein, text, etc.)
            trust_remote_code: Whether to trust remote code in model loading
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_type = model_type
        self.trust_remote_code = trust_remote_code
        
        # Load the model and tokenizer
        self.model, self.tokenizer = self._load_model()
        self.layer_info = self._extract_layer_info()
        
    def _load_model(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_id}")
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=self.trust_remote_code
            )
            print(f"✓ Tokenizer loaded successfully")
            print(f"  Vocabulary size: {tokenizer.vocab_size}")
            
            # Load model - try different approaches for custom models
            print("Loading model...")
            model = None
            
            # First, try AutoModelForCausalLM for language models
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                )
                print("✓ Model loaded as AutoModelForCausalLM")
            except Exception as e1:
                print(f"⚠ AutoModelForCausalLM failed: {e1}")
                
                # Try AutoModel as fallback
                try:
                    model = AutoModel.from_pretrained(
                        self.model_id, 
                        trust_remote_code=self.trust_remote_code,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    )
                    print("✓ Model loaded as AutoModel")
                except Exception as e2:
                    print(f"⚠ AutoModel failed: {e2}")
                    
                    # Try with device_map for large models
                    try:
                        model = AutoModel.from_pretrained(
                            self.model_id, 
                            trust_remote_code=self.trust_remote_code,
                            device_map="auto" if torch.cuda.is_available() else None,
                            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                        )
                        print("✓ Model loaded with device_map")
                    except Exception as e3:
                        print(f"⚠ AutoModel with device_map failed: {e3}")
                        
                        # Try loading the model class directly for custom architectures
                        try:
                            from transformers import AutoConfig
                            config = AutoConfig.from_pretrained(
                                self.model_id, 
                                trust_remote_code=self.trust_remote_code
                            )
                            
                            # Get the model class from the config
                            model_class = config.auto_map.get("AutoModelForCausalLM", None)
                            if model_class is None:
                                model_class = config.auto_map.get("AutoModel", None)
                            
                            if model_class is not None:
                                # Import the model class dynamically
                                import importlib
                                module_path, class_name = model_class.split(".")
                                module = importlib.import_module(f"transformers_modules.{self.model_id.replace('/', '.')}.{module_path}")
                                model_class_obj = getattr(module, class_name)
                                
                                model = model_class_obj.from_pretrained(
                                    self.model_id,
                                    trust_remote_code=self.trust_remote_code,
                                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                                )
                                print(f"✓ Model loaded using custom class: {class_name}")
                            else:
                                raise RuntimeError("No suitable model class found in auto_map")
                                
                        except Exception as e4:
                            print(f"❌ All model loading methods failed:")
                            print(f"  AutoModelForCausalLM: {e1}")
                            print(f"  AutoModel: {e2}")
                            print(f"  AutoModel with device_map: {e3}")
                            print(f"  Custom model class: {e4}")
                            raise e4
            
            if model is None:
                raise RuntimeError("Failed to load model with any method")
            
            # Move to GPU if available and not already on GPU
            if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
                model = model.to("cuda")
                print("✓ Model moved to GPU")
            elif torch.cuda.is_available():
                print("✓ Model already on GPU")
            else:
                print("⚠ CUDA not available, using CPU")
            
            model.eval()
            print(f"✓ Model loaded successfully")
            print(f"  Model type: {type(model).__name__}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _extract_layer_info(self) -> Dict:
        """Extract information about all model layers."""
        print("Extracting layer information...")
        
        layer_info = {}
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                try:
                    # Handle BFloat16 and other data types for statistics
                    if param.data.dtype == torch.bfloat16:
                        # Convert BFloat16 to Float32 for statistics
                        data_float = param.data.float()
                        mean_val = data_float.mean().item()
                        std_val = data_float.std().item()
                        min_val = data_float.min().item()
                        max_val = data_float.max().item()
                    else:
                        mean_val = param.data.mean().item()
                        std_val = param.data.std().item()
                        min_val = param.data.min().item()
                        max_val = param.data.max().item()
                    
                    layer_info[name] = {
                        'shape': list(param.shape),
                        'num_params': param.numel(),
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'dtype': str(param.dtype),
                        'device': str(param.device)
                    }
                    total_params += param.numel()
                except Exception as e:
                    print(f"Warning: Could not process statistics for {name}: {e}")
                    # Still add basic info even if statistics fail
                    layer_info[name] = {
                        'shape': list(param.shape),
                        'num_params': param.numel(),
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'dtype': str(param.dtype),
                        'device': str(param.device)
                    }
                    total_params += param.numel()
        
        print(f"Found {len(layer_info)} trainable layers")
        print(f"Total parameters: {total_params:,}")
        
        return layer_info
    
    def plot_weight_distributions(self, save_plots: bool = True) -> None:
        """Plot weight distributions for all layers."""
        print("Creating weight distribution plots...")
        
        # Group layers by type
        layer_groups = {}
        for name, info in self.layer_info.items():
            # Extract layer type from name
            if 'embedding' in name.lower():
                layer_type = 'embedding'
            elif 'attention' in name.lower() or 'attn' in name.lower():
                layer_type = 'attention'
            elif 'mlp' in name.lower() or 'feed_forward' in name.lower() or 'ffn' in name.lower():
                layer_type = 'mlp'
            elif 'norm' in name.lower() or 'layer_norm' in name.lower():
                layer_type = 'normalization'
            elif 'output' in name.lower():
                layer_type = 'output'
            else:
                layer_type = 'other'
            
            if layer_type not in layer_groups:
                layer_groups[layer_type] = []
            layer_groups[layer_type].append((name, info))
        
        # Create subplots for each layer type
        n_types = len(layer_groups)
        n_cols = 2
        n_rows = (n_types + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (layer_type, layers) in enumerate(layer_groups.items()):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            
            # Collect all weights for this layer type
            all_weights = []
            for name, info in layers:
                # Get actual weight data
                for param_name, param in self.model.named_parameters():
                    if param_name == name:
                        try:
                            # Handle BFloat16 and other data types
                            if param.data.dtype == torch.bfloat16:
                                # Convert BFloat16 to Float32 for processing
                                weights = param.data.float().flatten().cpu().numpy()
                            else:
                                weights = param.data.flatten().cpu().numpy()
                            
                            # Limit to reasonable number of weights for performance
                            if len(weights) > 10000:
                                weights = np.random.choice(weights, 10000, replace=False)
                            all_weights.extend(weights)
                        except Exception as e:
                            print(f"Warning: Could not process weights for {name}: {e}")
                        break
            
            if all_weights and len(all_weights) > 0:
                try:
                    ax.hist(all_weights, bins=min(50, len(all_weights)//10), alpha=0.7, density=True)
                    ax.set_title(f"{layer_type.title()} Layer Weights")
                    ax.set_xlabel("Weight Value")
                    ax.set_ylabel("Density")
                    ax.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Warning: Could not plot histogram for {layer_type}: {e}")
                    ax.text(0.5, 0.5, f"Error plotting {layer_type}", 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f"No weights for {layer_type}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide empty subplots
        for idx in range(n_types, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)
        
        plt.suptitle(f"Weight Distributions - {self.model_id}", fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / "weight_distributions.png", dpi=300, bbox_inches='tight')
            print(f"Saved weight distributions to: {self.output_dir / 'weight_distributions.png'}")
        
        plt.close()
    
    def plot_layer_statistics(self, save_plots: bool = True) -> None:
        """Plot layer statistics."""
        print("Creating layer statistics plots...")
        
        # Prepare data
        names = list(self.layer_info.keys())
        means = [info['mean'] for info in self.layer_info.values()]
        stds = [info['std'] for info in self.layer_info.values()]
        num_params = [info['num_params'] for info in self.layer_info.values()]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean weights by layer
        axes[0, 0].bar(range(len(names)), means)
        axes[0, 0].set_title("Mean Weight Values by Layer")
        axes[0, 0].set_xlabel("Layer Index")
        axes[0, 0].set_ylabel("Mean Weight Value")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Standard deviation by layer
        axes[0, 1].bar(range(len(names)), stds)
        axes[0, 1].set_title("Weight Standard Deviation by Layer")
        axes[0, 1].set_xlabel("Layer Index")
        axes[0, 1].set_ylabel("Weight Std")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Number of parameters by layer
        axes[1, 0].bar(range(len(names)), num_params)
        axes[1, 0].set_title("Number of Parameters by Layer")
        axes[1, 0].set_xlabel("Layer Index")
        axes[1, 0].set_ylabel("Number of Parameters")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Weight range (min to max) by layer
        mins = [info['min'] for info in self.layer_info.values()]
        maxs = [info['max'] for info in self.layer_info.values()]
        ranges = [max_val - min_val for min_val, max_val in zip(mins, maxs)]
        
        axes[1, 1].bar(range(len(names)), ranges)
        axes[1, 1].set_title("Weight Range (Max - Min) by Layer")
        axes[1, 1].set_xlabel("Layer Index")
        axes[1, 1].set_ylabel("Weight Range")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / "layer_statistics.png", dpi=300, bbox_inches='tight')
            print(f"Saved layer statistics to: {self.output_dir / 'layer_statistics.png'}")
        
        plt.close()
    
    def create_attention_visualization(self, sequence: str = None, save_plots: bool = True) -> None:
        """Create attention pattern visualization using real attention weights."""
        print("Creating attention pattern visualization...")
        
        # Use default sequence if not provided
        if sequence is None:
            if self.model_type == "protein":
                sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
            else:
                # Use a longer sequence for better attention visualization
                sequence = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more context for attention pattern analysis. We need more tokens to see meaningful attention patterns."
        
        # Tokenize sequence
        inputs = self.tokenizer(sequence, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        seq_len = inputs['input_ids'].shape[1]
        
        # Get token texts for labeling
        token_ids = inputs['input_ids'][0].cpu().numpy()
        token_texts = [self.tokenizer.decode([token_id]) for token_id in token_ids]
        # Truncate long token texts for display
        token_labels = [text[:8] + "..." if len(text) > 8 else text for text in token_texts]
        
        print(f"Sequence length: {seq_len} tokens")
        print(f"Tokens: {token_texts}")
        
        # Try to get real attention weights
        attention_weights = self._extract_real_attention_weights(inputs)
        
        if attention_weights is None:
            print("Could not extract real attention weights. This model may not support attention output.")
            return
        
        print(f"Extracted attention weights from {len(attention_weights)} layers")
        
        # Create visualization
        n_layers = len(attention_weights)
        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        # Make the figure larger for better visibility
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (layer_name, attn_weights) in enumerate(attention_weights):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            
            # Plot attention heatmap
            im = ax.imshow(attn_weights, cmap='Blues', aspect='auto')
            ax.set_title(f"Layer {idx + 1}: {layer_name}", fontsize=10)
            ax.set_xlabel("Key Position", fontsize=9)
            ax.set_ylabel("Query Position", fontsize=9)
            
            # Add token labels if sequence is not too long
            if seq_len <= 20:  # Only add labels for shorter sequences
                ax.set_xticks(range(seq_len))
                ax.set_yticks(range(seq_len))
                ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(token_labels, fontsize=8)
            else:
                # For longer sequences, just show every few tokens
                step = max(1, seq_len // 10)
                ax.set_xticks(range(0, seq_len, step))
                ax.set_yticks(range(0, seq_len, step))
                ax.set_xticklabels([token_labels[i] for i in range(0, seq_len, step)], 
                                 rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels([token_labels[i] for i in range(0, seq_len, step)], 
                                 fontsize=8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide empty subplots
        for idx in range(n_layers, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)
        
        plt.suptitle(f"Real Attention Patterns - {self.model_id}", fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / "attention_patterns.png", dpi=300, bbox_inches='tight')
            print(f"Saved attention patterns to: {self.output_dir / 'attention_patterns.png'}")
        
        plt.close()
    
    def _extract_real_attention_weights(self, inputs) -> list:
        """Extract real attention weights from the model."""
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Try to get attention weights by setting output_attentions=True
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention_weights = []
                
                # Process each layer's attention weights
                for layer_idx, attn in enumerate(outputs.attentions):
                    # Handle BFloat16 attention weights
                    if attn.dtype == torch.bfloat16:
                        attn = attn.float()
                    
                    # attn shape is typically (batch_size, num_heads, seq_len, seq_len)
                    if len(attn.shape) == 4:
                        # Average across heads and batch
                        attn_avg = attn[0].mean(dim=0).cpu().numpy()  # (seq_len, seq_len)
                    elif len(attn.shape) == 3:
                        # Already averaged or single head
                        attn_avg = attn[0].cpu().numpy()  # (seq_len, seq_len)
                    else:
                        print(f"Unexpected attention shape: {attn.shape}")
                        continue
                    
                    attention_weights.append((f"Layer_{layer_idx}", attn_avg))
                
                return attention_weights
            else:
                print("Model does not return attention weights with output_attentions=True")
                return None
                
        except Exception as e:
            print(f"Error extracting attention weights: {e}")
            print("Trying alternative method...")
            
            # Alternative method: Hook into attention layers
            return self._extract_attention_with_hooks(inputs)
    
    def _extract_attention_with_hooks(self, inputs) -> list:
        """Extract attention weights using forward hooks."""
        attention_weights = []
        
        def attention_hook(module, input, output):
            # This is a simplified hook - real implementation would depend on model architecture
            if hasattr(output, 'attentions') and output.attentions is not None:
                for layer_idx, attn in enumerate(output.attentions):
                    if len(attn.shape) == 4:
                        attn_avg = attn[0].mean(dim=0).cpu().numpy()
                    else:
                        attn_avg = attn[0].cpu().numpy()
                    attention_weights.append((f"Layer_{layer_idx}", attn_avg))
        
        # Register hooks on attention modules
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return attention_weights if attention_weights else None
    
    
    
    def create_model_architecture_plot(self, save_plots: bool = True) -> None:
        """Create a plot showing model architecture overview."""
        print("Creating model architecture plot...")
        
        # Group layers by type
        layer_groups = {}
        for name, info in self.layer_info.items():
            if 'embedding' in name.lower():
                layer_type = 'embedding'
            elif 'attention' in name.lower() or 'attn' in name.lower():
                layer_type = 'attention'
            elif 'mlp' in name.lower() or 'feed_forward' in name.lower() or 'ffn' in name.lower():
                layer_type = 'mlp'
            elif 'norm' in name.lower() or 'layer_norm' in name.lower():
                layer_type = 'normalization'
            elif 'output' in name.lower():
                layer_type = 'output'
            else:
                layer_type = 'other'
            
            if layer_type not in layer_groups:
                layer_groups[layer_type] = []
            layer_groups[layer_type].append((name, info))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Number of layers by type
        layer_types = list(layer_groups.keys())
        layer_counts = [len(layers) for layers in layer_groups.values()]
        
        bars1 = ax1.bar(layer_types, layer_counts)
        ax1.set_title("Number of Layers by Type")
        ax1.set_ylabel("Number of Layers")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars1, layer_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # Plot 2: Parameters by layer type
        param_counts = [sum(info['num_params'] for _, info in layers) 
                       for layers in layer_groups.values()]
        
        bars2 = ax2.bar(layer_types, param_counts)
        ax2.set_title("Parameters by Layer Type")
        ax2.set_ylabel("Number of Parameters")
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')
        
        # Add value labels on bars
        for bar, count in zip(bars2, param_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f"{count:,}", ha='center', va='bottom', rotation=90)
        
        plt.suptitle(f"Model Architecture Overview - {self.model_id}", fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / "architecture_overview.png", dpi=300, bbox_inches='tight')
            print(f"Saved architecture overview to: {self.output_dir / 'architecture_overview.png'}")
        
        plt.close()
    
    def save_model_summary(self) -> None:
        """Save detailed model summary to JSON."""
        print("Saving model summary...")
        
        total_params = sum(info['num_params'] for info in self.layer_info.values())
        
        summary = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "total_parameters": total_params,
            "total_layers": len(self.layer_info),
            "vocabulary_size": self.tokenizer.vocab_size,
            "cuda_available": torch.cuda.is_available(),
            "model_config": {
                "hidden_size": getattr(self.model.config, 'hidden_size', 'unknown'),
                "num_layers": getattr(self.model.config, 'num_hidden_layers', 'unknown'),
                "num_attention_heads": getattr(self.model.config, 'num_attention_heads', 'unknown'),
                "intermediate_size": getattr(self.model.config, 'intermediate_size', 'unknown'),
            },
            "layer_details": self.layer_info
        }
        
        with open(self.output_dir / "model_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved model summary to: {self.output_dir / 'model_summary.json'}")
    
    def run_full_visualization(self, sequence: str = None) -> None:
        """Run all visualization methods."""
        print("=" * 60)
        print(f"Model Visualization - {self.model_id}")
        print("=" * 60)
        
        total_params = sum(info['num_params'] for info in self.layer_info.values())
        print(f"Total Parameters: {total_params:,}")
        print(f"Total Layers: {len(self.layer_info)}")
        print(f"Vocabulary Size: {self.tokenizer.vocab_size}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print()
        
        # Generate all visualizations
        try:
            self.plot_weight_distributions()
        except Exception as e:
            print(f"⚠️  Error in weight distributions: {e}")
        
        try:
            self.plot_layer_statistics()
        except Exception as e:
            print(f"⚠️  Error in layer statistics: {e}")
        
        try:
            self.create_attention_visualization(sequence)
        except Exception as e:
            print(f"⚠️  Error in attention visualization: {e}")
        
        try:
            self.create_model_architecture_plot()
        except Exception as e:
            print(f"⚠️  Error in architecture plot: {e}")
        
        try:
            self.save_model_summary()
        except Exception as e:
            print(f"⚠️  Error saving model summary: {e}")
        
        print("=" * 60)
        print("Visualization Complete!")
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 60)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize any Hugging Face model")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--output_dir", type=str, default="./model_visualizations", help="Output directory")
    parser.add_argument("--model_type", type=str, default="auto", help="Model type (auto, protein, text)")
    parser.add_argument("--sequence", type=str, help="Input sequence for attention visualization")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code in model loading")
    
    args = parser.parse_args()
    
    try:
        visualizer = ModelVisualizer(
            model_id=args.model_id,
            output_dir=args.output_dir,
            model_type=args.model_type,
            trust_remote_code=args.trust_remote_code
        )
        visualizer.run_full_visualization(args.sequence)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
