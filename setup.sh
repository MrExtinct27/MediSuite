#!/bin/bash

# filepath: /home/ali/code/hackathon/MediSuite-Ai-Agent/setup.sh

# Update package list
echo "Updating package list..."
sudo apt update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y tesseract-ocr poppler-utils

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify installations
echo "Verifying installations..."
if command -v tesseract >/dev/null 2>&1; then
    echo "Tesseract OCR installed successfully."
else
    echo "Error: Tesseract OCR installation failed."
fi

if command -v pdfinfo >/dev/null 2>&1; then
    echo "Poppler installed successfully."
else
    echo "Error: Poppler installation failed."
fi

echo "Setup complete!"