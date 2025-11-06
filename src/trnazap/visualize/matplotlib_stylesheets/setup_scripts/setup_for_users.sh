#!/bin/bash
# setup_for_user.sh
# Setup script for Genometech Lab matplotlib plotting environment

echo "=============================================="
echo "Genometech Lab Plotting Setup"
echo "=============================================="

# Check if user is on the cluster
if [ ! -d "/projects/Genometechlab/matplotlib_stylesheets" ]; then
    echo "ERROR: Cannot find /projects/Genometechlab/matplotlib_stylesheets"
    echo "Are you on the Explorer cluster?"
    exit 1
fi

# Check if PYTHONPATH already configured
if grep -q "/projects/Genometechlab/matplotlib_stylesheets" ~/.bashrc 2>/dev/null; then
    echo "✓ PYTHONPATH already configured"
else
    echo "Adding PYTHONPATH to ~/.bashrc..."
    echo "" >> ~/.bashrc
    echo "# Genometech Lab Plotting Setup" >> ~/.bashrc
    echo 'export PYTHONPATH="${PYTHONPATH}:/projects/Genometechlab/matplotlib_stylesheets"' >> ~/.bashrc
    echo "✓ Added PYTHONPATH"
fi

# Check if MPLBACKEND already configured
if grep -q "MPLBACKEND" ~/.bashrc 2>/dev/null; then
    echo "✓ MPLBACKEND already configured"
else
    echo "Adding MPLBACKEND to ~/.bashrc..."
    echo "" >> ~/.bashrc
    echo "# Set matplotlib backend to PDF for non-interactive use" >> ~/.bashrc
    echo "# This prevents X11 connection warnings when SSH'd without -X/-Y" >> ~/.bashrc
    echo 'export MPLBACKEND=PDF' >> ~/.bashrc
    echo "✓ Added MPLBACKEND=PDF"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: source ~/.bashrc"
echo "2. Test: python -c 'from genometechlab_plotting import quick_test; quick_test()'"
echo ""
echo "Note: MPLBACKEND is set to PDF for non-interactive use."
echo "      This prevents X11 warnings when working over SSH."
echo "      To use interactive plots, you can override with:"
echo "      export MPLBACKEND=Qt5Agg  (or TkAgg)"
echo ""
echo "For help, see: /projects/Genometechlab/matplotlib_stylesheets/QUICK_START.txt"
echo "=============================================="