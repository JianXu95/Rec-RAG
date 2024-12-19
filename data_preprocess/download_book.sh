#!/bin/bash

# Set the URL for the dataset
URL="https://www.kaggle.com/api/v1/datasets/download/ruchi798/bookcrossing-dataset"

# Set the output filename
OUTPUT_FILE="book-crossing.zip"

# Directory to store the dataset
DATASET_DIR="../data/BookCrossing/raw_data"

# Check if wget or curl is installed
if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null
then
    echo "Please install wget or curl to download the dataset."
    exit 1
fi

# Create a directory for the dataset if it doesn't exist
if [ ! -d "$DATASET_DIR" ]; then
    mkdir -p "$DATASET_DIR"
fi

# Download the dataset using wget or curl
if command -v wget &> /dev/null
then
    echo "Downloading BX dataset using wget..."
    wget -O "$DATASET_DIR/$OUTPUT_FILE" "$URL"
elif command -v curl &> /dev/null
then
    echo "Downloading BX dataset using curl..."
    curl -o "$DATASET_DIR/$OUTPUT_FILE" "$URL"
fi

# Unzip the dataset
echo "Unzipping the dataset..."
unzip "$DATASET_DIR/$OUTPUT_FILE" -d "$DATASET_DIR"

# Remove the zip file after extraction
echo "Cleaning up..."
rm "$DATASET_DIR/$OUTPUT_FILE"

echo "Download and extraction complete!"
