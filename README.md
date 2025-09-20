# Model Visualization Library

A reusable library for visualizing weights and attention patterns of any Hugging Face transformer model.

## Features

- **Generic Model Support**: Works with any Hugging Face model
- **Docker-based**: Runs in isolated environment with GPU support
- **Comprehensive Visualizations**: Weight distributions, attention patterns, layer statistics
- **Easy to Use**: Simple command-line interface
- **Reusable Library**: Use directly in Python code

## Quick Start

### 1. Visualize any model with Docker

```bash
# Protein language model
./run_visualization.sh --model_id "nvidia/AMPLIFY_350M" --model_type protein

# Text model
./run_visualization.sh --model_id "bert-base-uncased" --model_type text

# With custom sequence
./run_visualization.sh --model_id "gpt2" --model_type text --sequence "Hello world"
```

### 2. Use the library directly in Python

```python
from model_visualizer import ModelVisualizer

# Create visualizer
visualizer = ModelVisualizer(
    model_id="nvidia/AMPLIFY_350M",
    output_dir="./my_visualizations",
    model_type="protein"
)

# Run all visualizations
visualizer.run_full_visualization()
```

### 3. Run example script

```bash
python example.py
```

## Files

- **`model_visualizer.py`** - Core visualization library
- **`visualize_model.py`** - Docker-compatible script
- **`run_visualization.sh`** - Easy-to-use runner script
- **`example.py`** - Example usage script
- **`requirements.txt`** - Dependencies
- **`Dockerfile`** - Docker configuration (optional)

## Generated Outputs

The visualization generates:

1. **`weight_distributions.png`** - Weight histograms by layer type
2. **`layer_statistics.png`** - Statistical analysis of layers
3. **`attention_patterns.png`** - Attention pattern visualization
4. **`architecture_overview.png`** - Model architecture breakdown
5. **`model_summary.json`** - Detailed model information

## Requirements

- Docker with GPU support
- NVIDIA Container Toolkit
- Internet connection for model download

## Examples

### Protein Models
```bash
./run_visualization.sh --model_id "nvidia/AMPLIFY_350M" --model_type protein
./run_visualization.sh --model_id "facebook/esm2_t6_8M_UR50D" --model_type protein
```

### Text Models
```bash
./run_visualization.sh --model_id "bert-base-uncased" --model_type text
./run_visualization.sh --model_id "gpt2" --model_type text --sequence "The quick brown fox"
```

### Custom Models
```bash
./run_visualization.sh --model_id "your-username/your-model" --trust_remote_code
```

## Troubleshooting

- **GPU not available**: The script will fall back to CPU
- **Model download fails**: Check internet connection and model ID
- **Permission errors**: Ensure Docker has proper permissions
- **Memory issues**: Try with smaller models first