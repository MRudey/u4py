"""Testing custom colormap as requested by HLNUG"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Data generation
x = np.linspace(0, 5, 400)
y = np.linspace(0, 5, 400)
X, Y = np.meshgrid(x, y)
Z = (np.sin(X) * np.cos(Y) + 1) * 45  # Values between 0 and 90

# Color thresholds and corresponding named colors
bounds = [0, 5, 10, 20, 30, 45, 60]
mapping = [b / 60 for b in bounds]
color_names = [
    "lightgreen",  # 0-5
    "green",  # 5-10
    "lightyellow",  # 10-20
    "yellow",  # 20-30
    "orange",  # 30-45
    "red",  # 45-60
    "darkviolet",  # 60-90
]
col_list = [(m, c) for m, c in zip(mapping, color_names)]
cmap = LinearSegmentedColormap.from_list("custom", colors=col_list)

fig, ax = plt.subplots()
img = ax.imshow(
    Z,
    cmap=cmap,
    vmin=0,
    vmax=60,
)
# Add colorbar
cbar = fig.colorbar(img, ax=ax, extend="max")
cbar.set_label("Slope")

fig.tight_layout()
fig.savefig("cmap_test.png")
