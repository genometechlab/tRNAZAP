# Genometech Lab Matplotlib Plotting - Quick Start Guide

## Initial Setup (One Time Only)

### 1. Run the setup script:
```bash
cd /projects/Genometechlab/matplotlib_stylesheets/setup_scripts
./setup_for_user.sh
source ~/.bashrc
```

### 2. Test it works:
```bash
python -c "from genometechlab_plotting import quick_test; quick_test()"
```

## Basic Usage

Add these two lines to any Python script:

```python
from genometechlab_plotting import setup_style
setup_style()
```

That's it! Now all your plots will use:
- **Helvetica fonts** (Bold for titles, Regular for labels)
- **Transparent backgrounds** (no white boxes in PDFs)
- **Clean style** (no top/right spines)
- **High resolution** (600 DPI)
- **Colorblind-friendly colors**
 
## Example Script

```python
from genometechlab_plotting import setup_style, get_colors
import matplotlib.pyplot as plt
import numpy as np

# Apply lab style
setup_style()

# Get lab colors
colors = get_colors()

# Create plot
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), color=colors[0], label='Data 1')
ax.plot(x, np.cos(x), color=colors[1], label='Data 2')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Signal (mV)')
ax.set_title('My Experiment Results')
ax.legend()

plt.savefig('my_figure.pdf')
```

## To Create Multi-Page PDFs

Your setup already supports multi-page PDFs! Just use:

```python
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('multi_page_report.pdf') as pdf:
    # Your figures automatically save with the PDF backend
    pdf.savefig(fig1)
    pdf.savefig(fig2)
```

## Available Styles (Future Addition)

```python
setup_style('main')        # Default style (6.5 x 4.5 inches)
setup_style('single')      # Single panel figures
setup_style('two_panel')   # Side-by-side panels
setup_style('four_panel')  # 2x2 grid
```

### Currently Available (offerings for the plot peasants)
Style for inline plotting, when using in Jupyter ensure that this is setup as:
```python
setup_style('inline') # Style for inline plotting
```
Include this command at the top of any plotting cell:
```python
%matplotlib inline
```


## Lab Color Palette

```python
colors = get_colors()      # Returns all 8 colors
colors = get_colors(3)     # Returns first 3 colors
```

### Color Order:
| Index | Color | Hex Code |
|-------|-------|----------|
| 0 | Violet | `#332288` |
| 1 | Green | `#117733` |
| 2 | Teal | `#44AA99` |
| 3 | Sky Blue | `#88CCEE` |
| 4 | Yellow | `#DDCC77` |
| 5 | Coral | `#CC6677` |
| 6 | Magenta | `#AA4499` |
| 7 | Mauve | `#882255` |

## Troubleshooting

### Import error?
- Make sure you ran: `source ~/.bashrc`
- Check PYTHONPATH: `echo $PYTHONPATH`

### Fonts not working?
- The system should load them automatically
- Check PDF properties to verify Helvetica is embedded

### Style not applying?
- Make sure `setup_style()` is called BEFORE creating figures
- Don't use `plt.style.use()` after `setup_style()`
- Make sure not to use plt.tight_layout() when saving

## Need Help?

**Contact:** Andrew Stein  
**Location:** `/projects/Genometechlab/matplotlib_stylesheets/`  

Link to info about the colorblind friendly palette: https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40