#!/bin/bash

# Start Ollama server in the background
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 10

# Pull the model (this runs once on first boot, cached after)
echo "Pulling llama3.2:3b model..."
ollama pull llama3.2:3b

echo "Model ready. Starting Streamlit..."

# Start Streamlit on port 7860 (required by HuggingFace Spaces)
streamlit run DeployApp.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true
