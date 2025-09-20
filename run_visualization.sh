#!/bin/bash

# Generic Model Visualization Runner
# Usage: ./run_visualization.sh --model_id "model_name" [options]

MODEL_ID=""
OUTPUT_DIR="./visualizations"
MODEL_TYPE="auto"
SEQUENCE=""
TRUST_REMOTE_CODE=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_id) MODEL_ID="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --sequence) SEQUENCE="$2"; shift ;;
        --trust_remote_code) TRUST_REMOTE_CODE="--trust_remote_code"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if model_id is provided
if [ -z "$MODEL_ID" ]; then
    echo "Error: --model_id is required"
    echo "Usage: $0 --model_id 'model_name' [--output_dir './output'] [--model_type 'auto'] [--sequence 'input_text'] [--trust_remote_code]"
    echo ""
    echo "Examples:"
    echo "  $0 --model_id 'nvidia/AMPLIFY_350M' --model_type protein"
    echo "  $0 --model_id 'bert-base-uncased' --model_type text"
    echo "  $0 --model_id 'gpt2' --model_type text --sequence 'Hello world'"
    exit 1
fi

echo "=========================================="
echo "Model Visualization - Docker"
echo "=========================================="
echo "Configuration:"
echo "  Model ID: $MODEL_ID"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Model Type: $MODEL_TYPE"
if [ -n "$SEQUENCE" ]; then
    echo "  Sequence: ${SEQUENCE:0:50}..." # Show first 50 chars
fi
echo "  Trust Remote Code: $([ -n "$TRUST_REMOTE_CODE" ] && echo "Yes" || echo "No")"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Running Docker container with GPU support..."
echo "This may take a few minutes to download the model and generate visualizations..."
echo ""

# Build the command
CMD_ARGS="--model_id \"$MODEL_ID\" --output_dir \"/workspace/$OUTPUT_DIR\" --model_type \"$MODEL_TYPE\""
if [ -n "$SEQUENCE" ]; then
    CMD_ARGS="$CMD_ARGS --sequence \"$SEQUENCE\""
fi
if [ -n "$TRUST_REMOTE_CODE" ]; then
    CMD_ARGS="$CMD_ARGS $TRUST_REMOTE_CODE"
fi

# Auto-add trust_remote_code for certain models
if [[ "$MODEL_ID" == *"AMPLIFY"* ]] || [[ "$MODEL_ID" == *"nvidia"* ]]; then
    CMD_ARGS="$CMD_ARGS --trust_remote_code"
fi

# Run the Docker container
docker run --gpus all --rm \
    -v "$(pwd)/$OUTPUT_DIR":/workspace/"$OUTPUT_DIR" \
    -v "$(pwd)/model_visualizer.py":/workspace/model_visualizer.py \
    -v "$(pwd)/visualize_model.py":/workspace/visualize_model.py \
    -v "$(pwd)/requirements.txt":/workspace/requirements.txt \
    nvcr.io/nvidia/pytorch:25.08-py3 /bin/bash -c "\
    pip install --no-cache-dir -r requirements.txt && \
    python visualize_model.py $CMD_ARGS \
    "

# Check the exit code of the docker run command
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Visualization completed successfully!"
    echo "=========================================="
    echo ""
    echo "Generated files in $OUTPUT_DIR:"
    ls -l "$OUTPUT_DIR"
    echo ""
    echo "You can view the visualizations by opening the PNG files in $OUTPUT_DIR"
    echo "The JSON file contains detailed model information and statistics."
else
    echo ""
    echo "=========================================="
    echo "❌ Visualization failed!"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above and try again."
    echo "Common issues:"
    echo "  - Network connectivity for model download"
    echo "  - GPU availability"
    echo "  - Docker permissions"
    echo "  - Model compatibility"
    exit 1
fi
