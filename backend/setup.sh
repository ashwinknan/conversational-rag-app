#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Inform the user
echo "Virtual environment created and dependencies installed."

# Start the backend server
echo "Starting the backend server..."
python3 main.py
