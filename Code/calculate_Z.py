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

Calculate Z value for each player at ball snap moment.
take data from tracking.csv and players.csv
gameId = 2022091200, playId = 467

Author: Xiangtian Dai   donktr17@gmail.com
Created: 10th Dec, 2024

'''


import pandas as pd
import numpy as np
import json
from influence import calculate_influence, multivariate_gaussian

def process_play_influence():
    """
    Process player influence for a specific play and save results to JSON.
    Calculates influence parameters and Z-values for each player at ball snap.
    
    Args:
        None
        
    Returns:
        None (Saves results to 'visualInf.json')
    """
    # Read CSV file
    df = pd.read_csv('tracking_week_1.csv')
    
    # Filter specific game and play
    play_df = df[(df['gameId'] == 2022091200) & 
                 (df['playId'] == 467)]
    
    # Get data at ball snap moment
    snap_df = play_df[play_df['event'] == 'ball_snap']
    
    # Get ball position
    ball_pos = snap_df[snap_df['displayName'] == 'football'][['x', 'y']].values[0]
    
    # Initialize result dictionary
    result_dict = {}
    
    # Process each player
    for _, player in snap_df[snap_df['displayName'] != 'football'].iterrows():
        nfl_id = player['nflId']
        x = player['x']
        y = player['y']
        s = player['s']
        # Convert angle from degrees to radians
        theta = np.deg2rad(player['o'])
        
        # Calculate influence parameters
        mu, cov = calculate_influence(x, y, theta, s, ball_pos)
        
        # Calculate Z value at player position
        player_pos = np.array([x, y])
        Z = multivariate_gaussian(player_pos, mu, cov)
        
        # Calculate influence radius R_i
        ball_distance = np.linalg.norm(ball_pos - player_pos)
        R_i = 4 + 6 * min(1, ball_distance / 25)
        
        # Store results
        result_dict[int(nfl_id)] = (float(Z), float(R_i))
    
    # Save to JSON file
    with open('visualInf.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

if __name__ == "__main__":
    process_play_influence()