"""
Genometech Lab Matplotlib Plotting Package
==========================================

A matplotlib style system for consistent, publication-ready figures
with Helvetica fonts and lab-standard formatting.

Basic usage:
    from genometechlab_plotting import setup_style
    setup_style()
"""

# Import main functions from the module
from .genometechlab_plotting import setup_style, get_colors, create_figure, quick_test

# Define package version
__version__ = '1.0.0'

# Define what's available when someone does "from genometechlab_plotting import *"
__all__ = ['setup_style', 'get_colors', 'create_figure', 'quick_test']

# Package metadata
__author__ = 'Genometech Lab'
__email__ = 'stein.an@northeastern.edu'
__description__ = 'Lab-standard matplotlib styling with Helvetica fonts'
