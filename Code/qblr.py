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

Visualize influence area based on tracking data and player's influence range.
Using Gaussian distribution to calculate influence.

Author: Xiangtian Dai   donktr17@gmail.com
Created: 10th Dec, 2024

'''


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from PIL import Image

def main():
    # ----------------------------
    # 1. Load Files
    # ----------------------------
    tracking_file = 'tracking_week_1.csv'
    players_file  = 'players.csv'
    visualInf_file = 'visualInf.json'
    background_img = 'footfield.png'
    
    df_tracking = pd.read_csv(tracking_file)
    df_players  = pd.read_csv(players_file)
    with open(visualInf_file, 'r') as f:
        visual_data = json.load(f)  # Format: { "nflId1": [Z1, r1], "nflId2": [Z2, r2], ... }
    
    # ----------------------------
    # 2. Filter Required Data from Tracking (gameId=2022091200, playId=467, event='ball_snap')
    # ----------------------------
    df_ball_snap = df_tracking[
        (df_tracking['gameId'] == 2022091200) &
        (df_tracking['playId'] == 467) &
        (df_tracking['event'] == 'ball_snap')
    ][['nflId','x','y','club']]
    
    # Coordinate adjustment: x+5, y+4
    df_ball_snap['x'] = df_ball_snap['x'] + 4
    df_ball_snap['y'] = df_ball_snap['y'] + 4

    # ----------------------------
    # 3. Merge with players.csv to get player positions
    # ----------------------------
    df_ball_snap = pd.merge(df_ball_snap,
                            df_players[['nflId','position']],
                            on='nflId',
                            how='left')
    
    # Find QB's club (offensive team)
    df_qb = df_ball_snap[df_ball_snap['position'] == 'QB']
    if len(df_qb) > 0:
        qb_club = df_qb['club'].iloc[0]  # Assuming only one QB
    else:
        qb_club = None

    # ----------------------------
    # 4. Create 2D Grid (1224x616) and Calculate Z_map (Gaussian Addition)
    # ----------------------------
    width_px, height_px = 1224, 616
    max_x, max_y = 122.4, 61.6
    
    grid_x = np.linspace(0, max_x, width_px)
    grid_y = np.linspace(0, max_y, height_px)
    gx, gy = np.meshgrid(grid_x, grid_y)  # gy.shape -> (height_px, width_px)

    Z_map = np.zeros_like(gx, dtype=float)  # Initialize with zeros

    # Add Gaussian distribution for each player (nflId -> [Z_i, r_i])
    for nflId_str, (Z_i, r_i) in visual_data.items():
        # Convert JSON string key to int
        try:
            nflId_num = int(nflId_str)
        except:
            # Skip if conversion fails
            continue
        
        # Find player coordinates in df_ball_snap
        row = df_ball_snap[df_ball_snap['nflId'] == nflId_num]
        if row.empty:
            continue
        
        x_i = row['x'].values[0]
        y_i = row['y'].values[0]
        club = row['club'].values[0]
        
        # Determine if the player is on the offensive or defensive team
        if qb_club is not None and club == qb_club:
            Z_i = -abs(Z_i)  # Offensive team: make Z negative
        else:
            Z_i = abs(Z_i)   # Defensive team: make Z positive
        
        # Calculate Gaussian distribution contribution
        dist_sq = (gx - x_i)**2 + (gy - y_i)**2
        contribution = Z_i * np.exp(- dist_sq / (2 * r_i**2))
        
        # Apply cutoff for distances greater than r_i
        contribution[dist_sq > r_i**2] = 0
        
        Z_map += contribution

    # ----------------------------
    # 5. Custom RGBA based on Z value ranges
    # ----------------------------
    color_image = np.zeros((height_px, width_px, 4), dtype=float)  # RGBA

    # Condition masks
    mask_eq0  = (Z_map == 0)
    mask_p1   = (Z_map > 0) & (Z_map < 0.01)
    mask_p2   = (Z_map >= 0.01) & (Z_map < 0.02)
    mask_p3   = (Z_map >= 0.02)
    mask_n1   = (Z_map < 0) & (Z_map > -0.01)
    mask_n2   = (Z_map <= -0.01) & (Z_map > -0.02)
    mask_n3   = (Z_map <= -0.02)

    # Z=0 => alpha=0 (fully transparent)
    color_image[mask_eq0, 3] = 0.0

    # 0<Z<0.01 => (22, 200, 245, 0.4)
    color_image[mask_p1] = [22/255., 200/255., 245/255., 0.4]

    # 0.01<=Z<0.02 => (22, 118, 245, 0.4)
    color_image[mask_p2] = [22/255., 118/255., 245/255., 0.4]

    # Z>=0.02 => (22, 118, 245, 0.4)
    color_image[mask_p3] = [22/255., 118/255., 245/255., 0.4]

    # -0.01<Z<0 => (245, 245, 22, 0.4)
    color_image[mask_n1] = [245/255., 245/255., 22/255., 0.4]

    # -0.02<Z<=-0.01 => (255, 255, 135, 0.4)
    color_image[mask_n2] = [255/255., 255/255., 135/255., 0.4]

    # Z<=-0.02 => (255, 255, 255, 0.4)
    color_image[mask_n3] = [255/255., 255/255., 255/255., 0.4]

    # ----------------------------
    # 6. Plot Generation
    # ----------------------------
    fig, ax = plt.subplots(figsize=(12.24, 6.16))  # Multiply by dpi=100 for 1224x616 pixels
    
    # (a) Plot background image footfield.png
    img = Image.open(background_img)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    ax.imshow(img, extent=[0, max_x, 0, max_y], alpha=0.8, origin='lower')
    
    # (b) Plot color distribution (with alpha channel)
    ax.imshow(color_image, extent=[0, max_x, 0, max_y], origin='lower')
    
    # (c) Plot player positions as scatter points
    for idx, row in df_ball_snap.iterrows():
        px = row['x']
        py = row['y']
        pos = str(row['position'])
        club = row['club']
        
        # Marker shape
        if pos == 'QB':
            marker_style = '*'
        elif pos in ['DT', 'DE', 'C', 'G', 'T', 'NT']:
            if qb_club is not None and club == qb_club:
                marker_style = '^'  # Offensive: upward triangle
            else:
                marker_style = (3, 0, 0)  # Defensive: equilateral triangle
        else:
            marker_style = 'o'
        
        # Fill color
        if qb_club is not None and club == qb_club:
            color_fill = 'black'  # Offensive team
        else:
            color_fill = 'red'    # Defensive team
        
        # Draw scatter point
        ax.scatter(px, py, marker=marker_style, s=200, c=color_fill)

    # Set axis ranges
    ax.set_xlim([0, max_x])
    ax.set_ylim([0, max_y])
    ax.set_xlabel('X (yards)')
    ax.set_ylabel('Y (yards)')
    ax.set_title('Area of Influence Map')
    
    # ----------------------------
    # 7. Save Result
    # ----------------------------
    plt.savefig('qb_visual.jpg', dpi=100, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()
