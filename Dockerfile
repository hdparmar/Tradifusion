# Use the TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y git

# Install Python dependencies
RUN pip install --upgrade \
    'git+https://github.com/keras-team/keras-cv.git' \
    tensorflow \
    tensorflow_addons \
    'git+https://github.com/costiash/stable-diffusion-tensorflow' \
    pytorch-lightning \
    pydub \
    huggingface-hub \
    scikit-learn \
    gradio \
    ftfy \
    transformers \
    'git+https://github.com/huggingface/diffusers.git' \
    'git+https://github.com/riffusion/riffusion'

# Set up environment variable for dataset path (can be overridden at runtime)
ENV DATASET_PATH=/workspace/dataset

# Expose the port for Jupyter
EXPOSE 8888

# Set the working directory
WORKDIR /workspace

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]

