#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv v_env
source v_env/bin/activate

if [ -d "v_env/bin" ] && [ -f "v_env/pyvenv.cfg" ]; then
	echo "Virtual environment was created successfully."
else
	echo "Virtual environment was not created."
	exit 1
fi

echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt
