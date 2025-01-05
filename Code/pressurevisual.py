'''
The MIT License (MIT)

CopyRight (c) 2024-2025 Xiangtian Dai

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
    and associated documentation files (the “Software”), to deal in the Software without restriction, 
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial 
    portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,  OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Visualize pressure value based on defender's position and generate animation.

Author: Xiangtian Dai   donktr17@gmail.com
Created: 10th Dec, 2024

'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Wedge, Circle


# Isosceles triangle vertex (apex pointing right)
apex_x, apex_y = 17, 17

# Triangle side length and base length
side_length = 1
base_length = 0.5

# Calculate triangle height using geometric relationship
# height = sqrt(side_length^2 - (base_length/2)^2)
h = np.sqrt(side_length**2 - (base_length * 0.5)**2)

# Base point at (17,17), base center is h units left of the apex
# Base center coordinates
base_center = (apex_x - h, apex_y)
# Upper and lower base endpoints
upper_base = (base_center[0], base_center[1] + base_length/2)
lower_base = (base_center[0], base_center[1] - base_length/2)



green_radius = 4.37
blue_radius = 2.18
yellow_radius = 3.28

# Green sector range
green_theta1, green_theta2 = -70, 70
# Blue sector range
blue_theta1, blue_theta2   = 135, 225

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal', adjustable='box')

# Draw triangle
triangle = Polygon([(apex_x, apex_y),
                   upper_base,
                   lower_base],
                  closed=True,
                  fill=False,
                  edgecolor='black',
                  linewidth=2)
ax.add_patch(triangle)

# Draw green sector
green_wedge = Wedge(center=(apex_x, apex_y), 
                    r=green_radius,
                    theta1=green_theta1,
                    theta2=green_theta2,
                    facecolor='green', alpha=0.6)
ax.add_patch(green_wedge)

# Draw blue sector
blue_wedge = Wedge(center=(apex_x, apex_y),
                   r=blue_radius,
                   theta1=blue_theta1,
                   theta2=blue_theta2,
                   facecolor='blue', alpha=0.6)
ax.add_patch(blue_wedge)

# Yellow sector 1: [70°, 135°]
yellow_wedge1 = Wedge(center=(apex_x, apex_y),
                      r=yellow_radius,
                      theta1=70, theta2=135,
                      facecolor='yellow', alpha=0.5)
ax.add_patch(yellow_wedge1)

# Yellow sector 2: [225°, 290°]
yellow_wedge2 = Wedge(center=(apex_x, apex_y),
                      r=yellow_radius,
                      theta1=225, theta2=290,
                      facecolor='yellow', alpha=0.5)
ax.add_patch(yellow_wedge2)

# Ball initial position and radius
ball_radius = 0.5
ball_x_init, ball_y_init = 22.0, 18.0

# Represent ball using Circle Patch
ball = Circle((ball_x_init, ball_y_init), 
              radius=ball_radius, 
              facecolor='red', 
              edgecolor='black')
ax.add_patch(ball)

# Set plot range to show ball movement from right to left
ax.set_xlim(10, 30)
ax.set_ylim(10, 25)

# Display pressure value text in upper right corner
# transform=ax.transAxes uses relative coordinates (0~1)
text_pressure = ax.text(0.95, 0.95, 
                        "Pressure: 0.00", 
                        transform=ax.transAxes,
                        ha="right", va="top",
                        fontsize=12, color='black')

def compute_pressure(x_ball, y_ball):
    """
    Calculate pressure based on ball position relative to sectors.
    
    Args:
        x_ball (float): x-coordinate of the ball
        y_ball (float): y-coordinate of the ball
        
    Returns:
        float: Pressure value (0~1) based on distance from apex and sector position.
               Returns 0 if outside all sectors or beyond sector radius.
    """
    # Calculate distance from ball to triangle apex
    dx = x_ball - apex_x
    dy = y_ball - apex_y
    d = np.sqrt(dx*dx + dy*dy)

    # Calculate angle with positive x-axis (0~360)
    # atan2 outputs (-180°,180°), needs adjustment
    angle_deg = np.degrees(np.arctan2(dy, dx))
    if angle_deg < 0:
        angle_deg += 360  # Adjust to [0, 360)

    # Check which sector contains the ball (check angle range first, then radius)

    # Green sector: [-70, 70], radius 4.37
    if -70 <= angle_deg <= 70 and d <= green_radius:
        return max(0, 1 - d/green_radius)

    # Blue sector: [135, 225], radius 2.18
    if 135 <= angle_deg <= 225 and d <= blue_radius:
        return max(0, 1 - d/blue_radius)

    # Yellow sectors: [70, 135] and [225, 290]
    if ((70 <= angle_deg <= 135) or 
        (225 <= angle_deg <= 290)) and d <= yellow_radius:
        return max(0, 1 - d/yellow_radius)

    return 0.0


# Ball moves 0.05 units per frame
frames_count = 200  # Adjustable based on needs
move_per_frame = 0.05

def update(frame):
    """
    Update function for animation frame.
    
    Args:
        frame (int): Current frame index
        
    Returns:
        tuple: Updated ball and pressure text objects
    """
    # Move ball leftward from (25, 20)
    x_new = ball_x_init - frame * move_per_frame
    y_new = ball_y_init

    # Update ball position
    ball.center = (x_new, y_new)

    # Calculate pressure
    pres = compute_pressure(x_new, y_new)

    # Update text content
    text_pressure.set_text(f"Pressure: {pres:.4f}")

    return ball, text_pressure


anim = FuncAnimation(fig, 
                     update, 
                     frames=frames_count, 
                     interval=50,   # milliseconds, adjustable for speed
                     blit=False)    # blit=False for proper text updates

# Save animation as GIF
anim.save('pressure.gif', writer=PillowWriter(fps=20))

plt.show()
