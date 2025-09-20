#!/usr/bin/env python3
"""
Example usage of the Model Visualizer library.

This script demonstrates how to use the ModelVisualizer class directly
without Docker for local development and testing.
"""

import os
import sys
from pathlib import Path

from model_visualizer import ModelVisualizer


def main():
    """Example usage of the ModelVisualizer library."""
    
    print("=" * 60)
    print("Model Visualizer Library - Example Usage")
    print("=" * 60)
    
    # Example 1: Visualize a text model
    print("\n1. Visualizing BERT model...")
    try:
        bert_visualizer = ModelVisualizer(
            model_id="bert-base-uncased",
            output_dir="./example_bert_output",
            model_type="text"
        )
        bert_visualizer.run_full_visualization("The quick brown fox jumps over the lazy dog.")
        print("✅ BERT visualization completed!")
    except Exception as e:
        print(f"❌ BERT visualization failed: {e}")
    
    # Example 2: Visualize a protein model (if available)
    print("\n2. Visualizing AMPLIFY model...")
    try:
        amplify_visualizer = ModelVisualizer(
            model_id="nvidia/AMPLIFY_350M",
            output_dir="./example_amplify_output",
            model_type="protein",
            trust_remote_code=True
        )
        amplify_visualizer.run_full_visualization("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
        print("✅ AMPLIFY visualization completed!")
    except Exception as e:
        print(f"❌ AMPLIFY visualization failed: {e}")
    
    # Example 3: Custom visualization with specific methods
    print("\n3. Custom visualization example...")
    try:
        custom_visualizer = ModelVisualizer(
            model_id="gpt2",
            output_dir="./example_gpt2_output",
            model_type="text"
        )
        
        # Run only specific visualizations
        custom_visualizer.plot_weight_distributions()
        custom_visualizer.plot_layer_statistics()
        custom_visualizer.create_attention_visualization("Hello world, this is a test.")
        custom_visualizer.save_model_summary()
        
        print("✅ Custom GPT-2 visualization completed!")
    except Exception as e:
        print(f"❌ Custom visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed! Check the output directories for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
