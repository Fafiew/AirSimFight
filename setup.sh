#!/bin/bash
# Setup script for AirSimFight

echo "AirSimFight Setup"
echo "================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create directories
echo "Creating directories..."
mkdir -p models logs scenarios config

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Make scripts executable
chmod +x examples/*.sh

echo ""
echo "Setup complete!"
echo ""
echo "To run pretraining:"
echo "  ./examples/run_pretrain.sh"
echo ""
echo "To run dual training:"
echo "  ./examples/run_dual.sh"
echo ""
echo "To visualize:"
echo "  python tools/visualize.py --scenario scenarios/example_visualization.json"
echo ""
echo "To run tests:"
echo "  pytest -q"
