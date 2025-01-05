'''
The MIT License (MIT)

CopyRight (c) 2024-2025 Xiangtian Dai

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
    and associated documentation files (the "Software"), to deal in the Software without restriction, 
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial 
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,  OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES 
    OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Just a simple program that make a table to see QBs' riskiness in terms of passes/handoff


Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Circle, Rectangle


# Data Preparation

teamACC = {'MIN': ["Kirk Cousins", 0.9149425287356322], 
'NYG': ["Daniel Jones", 0.9146005509641874], 
'LAC': ["Justin Herbert", 0.9021739130434783], 
'PIT': ["Kenny Pickett", 0.8997613365155132], 
'BAL': ["Lamar Jackson", 0.8960396039603961],
'DAL': ["Dak Prescott", 0.894484412470024], 
'NO': ["Andy Dalton", 0.8893805309734514], 
'JAX': ["Trevor Lawrence", 0.8870292887029289], 
'MIA': ["Tua Tagovailoa", 0.886021505376344],
'TEN': ["Ryan Tannehill", 0.8835227272727273]}


# Convert data to list and sort by risk score (1-acc) in descending order
data = []
for team, (qb, acc) in teamACC.items():
    risk = (1 - acc) * 5  #times 5 to get a better visualization
    data.append((qb, team, acc, risk))

# Sort by risk score in descending order
data.sort(key=lambda x: x[3], reverse=False)

# Add rankings
data = [(i, *row) for i, row in enumerate(data, start=1)]


FIG_WIDTH = 8           # Figure width (inches)
FIG_HEIGHT = 6          # Figure height (inches)
HEADER_HEIGHT = 0.08    # Header area height (relative coordinates)
ROW_HEIGHT = 0.07       # Height of each row (relative coordinates)
LEFT_MARGIN = 0.06      # Left margin
TOP_MARGIN = 0.90       # Table top starting position
FONT_SIZE = 10         # Font size

COL_X = {
    "Rank": 0.00,   # Start from left edge (will add LEFT_MARGIN later)
    "QBs":  0.08,   # Second column
    "Team": 0.45,   # Third column
    "Acc":  0.60,   # Fourth column, for circular progress bar
    "Risk": 0.80,   # Fifth column, for bar progress bar
}

# Colors
ODD_ROW_COLOR  = "#f0f0f0"
EVEN_ROW_COLOR = "#e0f6ff"


def draw_circle_gauge(ax, center, radius, fraction, color="tab:blue"):
    """
    Draw a circular progress gauge.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to draw on
        center (tuple): (x, y) coordinates of the circle center
        radius (float): Radius of the circle
        fraction (float): Fill fraction (between 0 and 1)
        color (str): Color of the progress gauge
    
    Returns:
        None
    """
    # Circle outline
    circle = Circle(center, radius, edgecolor='black', facecolor='none', lw=1)
    ax.add_patch(circle)
    
    # Fill wedge
    theta1 = 90            # Start from top
    theta2 = 90 - 360*fraction  # Rotate clockwise
    wedge = Wedge(center, radius, theta2, theta1, facecolor=color, alpha=0.6)
    ax.add_patch(wedge)

def draw_bar_gauge(ax, left, bottom, width, height, fraction, color="tab:green"):
    """
    Draw a horizontal bar progress gauge.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to draw on
        left (float): Left position of the bar
        bottom (float): Bottom position of the bar
        width (float): Width of the bar
        height (float): Height of the bar
        fraction (float): Fill fraction (between 0 and 1)
        color (str): Color of the progress bar
    
    Returns:
        None
    """
    # Draw border
    rect_border = plt.Rectangle((left, bottom), width, height,
                                edgecolor='black', facecolor='none', lw=1)
    ax.add_patch(rect_border)
    
    # Draw fill
    fill_w = width * fraction
    rect_fill = plt.Rectangle((left, bottom), fill_w, height,
                              edgecolor='none', facecolor=color, alpha=0.6)
    ax.add_patch(rect_fill)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Turn off axes (only use drawn elements)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Place header text
header_y = TOP_MARGIN
ax.text(LEFT_MARGIN + COL_X["Rank"], header_y, "Rank",
        ha="left", va="center", fontsize=FONT_SIZE, fontweight='bold')
ax.text(LEFT_MARGIN + COL_X["QBs"],  header_y, "QBs",
        ha="left", va="center", fontsize=FONT_SIZE, fontweight='bold')
ax.text(LEFT_MARGIN + COL_X["Team"], header_y, "Team",
        ha="left", va="center", fontsize=FONT_SIZE, fontweight='bold')
ax.text(LEFT_MARGIN + COL_X["Acc"],  header_y, "Team Accuracy",
        ha="center", va="center", fontsize=FONT_SIZE, fontweight='bold')
ax.text(LEFT_MARGIN + COL_X["Risk"], header_y, "Risk Score",
        ha="center", va="center", fontsize=FONT_SIZE, fontweight='bold')

for i, (rank, qb, team, acc, risk) in enumerate(data):
    # Calculate bottom y coordinate for this row
    row_y_bottom = TOP_MARGIN - (i+1)*ROW_HEIGHT
    
    # Fill odd/even row background
    row_color = ODD_ROW_COLOR if (i % 2 == 0) else EVEN_ROW_COLOR
    ax.add_patch(
        plt.Rectangle(
            (0, row_y_bottom),  # Bottom-left coordinate (start from x=0)
            1,                  # Width extends to x=1
            ROW_HEIGHT,         # Height
            facecolor=row_color,
            edgecolor='none',
            zorder=0
        )
    )
    
    # Write text (Rank, QBs, Team)
    text_y_center = row_y_bottom + ROW_HEIGHT/2
    
    ax.text(LEFT_MARGIN + COL_X["Rank"], text_y_center, f"{rank}",
            ha="left", va="center", fontsize=FONT_SIZE)
    
    ax.text(LEFT_MARGIN + COL_X["QBs"],  text_y_center, qb,
            ha="left", va="center", fontsize=FONT_SIZE)
    
    ax.text(LEFT_MARGIN + COL_X["Team"], text_y_center, team,
            ha="left", va="center", fontsize=FONT_SIZE)
    
    # Draw circular gauge + percentage (Team Accuracy)
    circle_center = (LEFT_MARGIN + COL_X["Acc"], text_y_center)
    circle_radius = 0.025  # Adjustable radius
    draw_circle_gauge(ax, circle_center, circle_radius, acc, color="tab:orange")
    # Write percentage in circle center with 1 decimal place
    ax.text(circle_center[0], circle_center[1],
            f"{acc*100:.1f}%",
            ha="center", va="center", fontsize=FONT_SIZE-1)
    
    # Draw bar gauge + percentage (Risk Score)
    bar_left   = LEFT_MARGIN + COL_X["Risk"] - 0.03
    bar_bottom = text_y_center - 0.015
    bar_width  = 0.1
    bar_height = 0.03
    draw_bar_gauge(ax, bar_left, bar_bottom, bar_width, bar_height, risk, color="tab:green")
    # Write percentage in bar center with 1 decimal place
    ax.text(bar_left + bar_width/2, text_y_center,
            f"{risk*100:.1f}%",
            ha="center", va="center", fontsize=FONT_SIZE-1, color="black")

plt.tight_layout()
plt.show()

# Save image to current directory
fig.savefig("risks_score.png", dpi=300, bbox_inches='tight')
print("Table has been generated and saved as risks_score.png")
