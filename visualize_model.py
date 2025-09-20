#!/usr/bin/env python3
"""
Docker-compatible Model Visualization Script

A simple script that uses the ModelVisualizer library to visualize any Hugging Face model.
"""

import os
import sys
from pathlib import Path

from model_visualizer import ModelVisualizer


def main():
    """Main function for Docker usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize any Hugging Face model in Docker")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--output_dir", type=str, default="./model_visualizations", help="Output directory")
    parser.add_argument("--model_type", type=str, default="auto", help="Model type (auto, protein, text)")
    parser.add_argument("--sequence", type=str, help="Input sequence for attention visualization")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code in model loading")
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("Docker Model Visualization")
        print("=" * 60)
        print(f"Model: {args.model_id}")
        print(f"Output: {args.output_dir}")
        print(f"Type: {args.model_type}")
        print("=" * 60)
        
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
