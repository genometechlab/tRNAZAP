#!/usr/bin/env python3
"""
genometechlab_plotting.py
=========================
Genometech Lab Plotting Module with Helvetica Font Management

This module provides consistent plotting styles for the lab with:
- Helvetica fonts (Bold for titles, Regular for text)
- Transparent backgrounds for easy figure editing
- Colorblind-friendly color palette
- Publication-ready settings
"""

# Font loading must happen before ANY matplotlib imports!
import warnings
from pathlib import Path

# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent.resolve()
FONTS_DIR = MODULE_DIR / 'fonts'
STYLES_DIR = MODULE_DIR / 'styles'

# Global variable to track if fonts are loaded
_FONTS_LOADED = False

def _load_fonts_early():
    """
    Load fonts before matplotlib is imported.
    This ensures Helvetica is available when matplotlib initializes.
    """
    global _FONTS_LOADED
    
    if _FONTS_LOADED:
        return True
    
    # Import only font_manager, not pyplot or full matplotlib
    import matplotlib.font_manager as fm
    
    if not FONTS_DIR.exists():
        warnings.warn(f"Fonts directory not found: {FONTS_DIR}")
        return False
    
    # Load all font files
    font_files = list(FONTS_DIR.glob('*.ttf')) + list(FONTS_DIR.glob('*.otf'))
    loaded_count = 0
    
    for font_path in font_files:
        try:
            fm.fontManager.addfont(str(font_path))
            loaded_count += 1
        except Exception as e:
            warnings.warn(f"Could not load {font_path.name}: {e}")
    
    if loaded_count > 0:
        # Clear font cache if method exists
        if hasattr(fm.fontManager, '_findfont_cached'):
            fm.fontManager._findfont_cached.cache_clear()
        
        print(f"Loaded {loaded_count} Helvetica fonts")
        _FONTS_LOADED = True
        return True
    else:
        warnings.warn("No fonts could be loaded!")
        return False

# Load Fonts before matplotlib is imported!
_load_fonts_early()

# Import matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Ensure Helvetica is prioritized
if _FONTS_LOADED:
    matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']

def setup_style(style='main', verbose=False):
    """
    Apply Genometech Lab plotting style.
    
    Parameters
    ----------
    style : str, optional
        Style name: 'main', 'single', 'two_panel', 'four_panel'
        Default is 'main'.
    verbose : bool, optional
        Whether to print status messages. Default is False.
    
    Examples
    --------
    >>> setup_style()  # Use default style
    >>> setup_style('two_panel')  # Use two-panel layout
    """
    # Map style names to style sheet files (Future Addition)
    style_map = {
        'main': 'genometechlab_main.mplstyle',
        'inline': 'genometechlab_inline.mplstyle',
        'two_panel': 'genometechlab_two_panel.mplstyle',
        'four_panel': 'genometechlab_four_panel.mplstyle',
    }
    
    if style not in style_map:
        raise ValueError(f"Unknown style: {style}. Choose from: {list(style_map.keys())}")
    
    # Apply the style sheet
    style_file = STYLES_DIR / style_map[style]
    if not style_file.exists():
        # If specific style doesn't exist, fall back to main
        if style != 'main':
            warnings.warn(f"Style '{style}' not found, using 'main' style")
            style_file = STYLES_DIR / style_map['main']
        else:
            raise FileNotFoundError(f"Main style file not found: {style_file}")
    
    plt.style.use(str(style_file))
    
    # Ensure Helvetica remains first in font list
    current_fonts = plt.rcParams['font.sans-serif']
    if 'Helvetica' not in current_fonts or current_fonts[0] != 'Helvetica':
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']
    
    if verbose:
        print(f"Applied style: {style}")
        print(f"Primary font: {plt.rcParams['font.sans-serif'][0]}")

def get_colors(n=None):
    """
    Get the Genometech Lab color palette.
    
    Parameters
    ----------
    n : int, optional
        Number of colors to return. If None, returns all 8 colors.
    
    Returns
    -------
    list
        List of color hex codes.
    
    Examples
    --------
    >>> colors = get_colors()  # Get all 8 colors
    >>> colors = get_colors(3)  # Get first 3 colors
    >>> ax.plot(x, y, color=colors[0])  # Use first color (blue)
    """
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', 
              '#F0E442', '#56B4E9', '#999999', '#000000']
    return colors[:n] if n else colors

def create_figure(title=None, style='main', figsize=None):
    """
    Create a figure with Genometech Lab style.
    
    Parameters
    ----------
    title : str, optional
        Figure title.
    style : str, optional
        Style to use. Default is 'main'.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    
    Examples
    --------
    >>> fig, ax = create_figure("My Experiment")
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> plt.show()
    """
    setup_style(style)
    
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()
    
    if title:
        ax.set_title(title)
    
    return fig, ax

def quick_test():
    """
    Run a quick test to verify the setup is working correctly.
    Creates test plots saved as quick_test.pdf and quick_test.png.
    
    Returns
    -------
    bool
        True if test successful, False otherwise.
    """
    print("\nRunning Genometech Lab plotting test...")
    
    try:
        # Apply style
        setup_style('main', verbose=True)
        
        # Create test plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        colors = get_colors(3)
        
        # Plot with lab colors
        for i, color in enumerate(colors[:3]):
            y = np.sin(x + i * np.pi/3)
            ax.plot(x, y, color=color, label=f'Dataset {i+1}', linewidth=2)
        
        # Labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal Amplitude')
        ax.set_title('Genometech Lab Style Test')
        ax.legend()
        
        # Save
        plt.savefig('quick_test.pdf')
        plt.close()
        
        print("\n✓ Test successful!")
        print("  Created: quick_test.pdf")
        print("\nTo verify Helvetica fonts are embedded:")
        print("  1. Open quick_test.pdf")
        print("  2. Check File > Properties > Fonts")
        print("  3. Should list Helvetica fonts")
        print("  4. Another option, open the image in Illustrator, then, navigate to:")
        print("     'Select', 'Object', 'All Text Objects'. This should show the font as Helvetica.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Make key functions available at module level
__all__ = ['setup_style', 'get_colors', 'create_figure', 'quick_test']

# Run quick test if executed as a script
if __name__ == '__main__':
    quick_test()