FROM nvcr.io/nvidia/pytorch:25.08-py3

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for visualization
RUN pip install --no-cache-dir \
    netron \
    tensorboard \
    seaborn \
    scikit-learn

# Copy the visualization scripts
COPY . .

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "visualize_amplify_weights.py"]
