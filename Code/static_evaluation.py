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

Generate a json file that contains initial evaluation for QB pass priority:
Throw the ball to the left side? right side?
To whom? RB? WR? TE?

Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''

import pandas as pd
import numpy as np
from math import radians, pi
import orjson
import glob
from influence import gaussian_influence_area
from multiprocessing import Pool, cpu_count
from functools import partial


def angle_to_radians_for_influence(o_deg):
    ''' 
    Regulate 0 degrees on y-positive axis

    Theta is 0 degrees on the y-positive axis, 
    and counterclockwise is the positive orientation

    Args: 
        o_deg: degree of orientation from tracking data.
    Return: 
        Corrected angle in radians as defined.
    '''
    return (pi / 2) - radians(o_deg)


def normalize_angle(angle):
    ''' 
    Normalize the angle to [-pi, pi]

    Args: 
        angle: numeric value

    Return: 
        angle, corrected version in [-pi, pi]
    '''
    return (angle + pi) % (2 * pi) - pi


def get_zone_pressure(distance, zone_radius):
    ''' 
    Calculate pressure = 1 - d/r_zone

    Args: 
        distance: Euclidean distance between two players
        zone_radius: zone radius for that zone

    Return: 
        pressure as a float
    '''
    if distance > zone_radius:
        return 0.0
    return 1 - distance / zone_radius


def compute_player_pressure(offensive_player, defensive_players):
    """ 
    Calculate overall pressure from all directions on an offensive player.

    Zones:
    - head-on: front ±70°, radius 4.37
    - hind: back ±45°, radius 2.18
    - lateral: remaining, radius 3.28

    Orientation (theta) is radian w.r.t. y-positive axis.

    Args: 
        offensive_player: dict with player info
        defensive_players: list of dicts for defensive players

    Return: 
        Rounded float pressure (4 decimals)
    """
    px, py = offensive_player['x'], offensive_player['y']
    theta = offensive_player['theta']

    pressure_sum = 0.0
    for dp in defensive_players:
        dx, dy = dp['x'], dp['y']
        vec = np.array([dx - px, dy - py])
        dist = np.linalg.norm(vec)
        if dist == 0:
            continue

        angle_vec = np.arctan2(vec[1], vec[0])
        angle_y_ref = angle_vec - pi / 2
        relative_angle = normalize_angle(angle_y_ref - theta)
        angle_deg = np.degrees(relative_angle)

        if -70 <= angle_deg <= 70:
            r_zone = 4.37
        elif abs(angle_deg) > 135:
            r_zone = 2.18
        else:
            r_zone = 3.28

        pressure_sum += get_zone_pressure(dist, r_zone)

    return round(float(pressure_sum), 4)


def get_eligible_receivers_side(players_sorted, side, LOS_x):
    ''' 
    Determine eligible receivers on a given side based on specified rules.
    Due to data precision, I have to consider an estimation approach to determine eligible receivers.
    On the LOS, it's hard to precisely say that a player just leave 0.01 yard away behind the line

    Args:
        players_sorted: list of player dicts sorted by y-coordinate.
        side: 'top' or 'bottom'.
        LOS_x: Line of Scrimmage x-coordinate.

    Return:
        A set of eligible receivers' nflIds.
    '''
    eligible = set()

    if not players_sorted:
        return eligible

    # Identify outermost player based on side
    if side == 'top':
        outermost = players_sorted[-1]
    else:
        outermost = players_sorted[0]

    # Rule 1: Check if outermost player is within 1 yard of LOS_x and has eligible position
    if abs(outermost['x'] - LOS_x) <= 0.75 and outermost['position'] in ['WR', 'TE', 'RB', 'FB']:
        eligible.add(outermost['nflId'])

    # Rule 2: Any offensive player beyond 1 yard from LOS_x and in [WR, TE, FB, RB] is eligible
    for player in players_sorted:
        if abs(player['x'] - LOS_x) > 0.75 and player['position'] in ['WR', 'TE', 'RB', 'FB']:
            eligible.add(player['nflId'])

    # Rule 3 is implicitly handled by not adding players who don't meet the above criteria

    return eligible


def process_tracking_file(file_path, plays_map, players_map, qb_positions):
    """
    Process a single tracking_week file to compute pressures and identify eligible receivers.

    Steps:
    - Load tracking data, filter ball_snap
    - For each play:
      - Identify offense/defense teams, ball position, QB, etc.
      - Compute influence area for all players
      - Determine eligible receivers based on positions and distance from LOS
      - Compute pressures only for eligible WR, TE, RB, FB
      - Sort by pressure ascending within each side
      - The side with larger influence sum (area) goes first in the final list, then the other side
      - Final output format: [{"nflId": pressure}, ...]
    
    Args:
        file_path: path to the tracking_week_*.csv file
        plays_map: dictionary mapping gamePlayId to play info
        players_map: dictionary mapping nflId to player info
        qb_positions: set of quarterback nflIds

    Return:
        A dictionary with gamePlayId as keys and list of {nflId: pressure} dicts as values
    """
    use_cols = ['gameId','playId','nflId','displayName','x','y','s','o','event','club','playDirection']
    dtype_map = {
        'gameId': 'int32',
        'playId': 'int32',
        'nflId': 'float32',
        'x': 'float32',
        'y': 'float32',
        's': 'float32',
        'o': 'float32',
        'club': 'object',
        'displayName': 'object',
        'event': 'object',
        'playDirection': 'object'
    }

    try:
        tracking_df = pd.read_csv(file_path, usecols=use_cols, dtype=dtype_map)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

    # Filter ball_snap
    tracking_df = tracking_df[tracking_df['event'] == 'ball_snap']

    if tracking_df.empty:
        return {}

    # Create unique gamePlayId
    tracking_df['gamePlayId'] = tracking_df['gameId'].astype(str) + "_" + tracking_df['playId'].astype(str)

    final_results = {}

    for gamePlayId, snap_group in tracking_df.groupby('gamePlayId'):
        if gamePlayId not in plays_map:
            continue
        info = plays_map[gamePlayId]

        if not isinstance(info, dict):
            print(f"Unexpected info type for {gamePlayId}: {type(info)}")
            continue

        pos_team = info['possessionTeam']
        def_team = info['defensiveTeam']
        yardline_num = info['yardlineNumber']

        # Get playDirection from tracking data
        play_dir = snap_group['playDirection'].iloc[0]  # 'left' or 'right'

        # LOS calculation: x = yardlineNumber + 10
        LOS_x = yardline_num + 10

        ball_data = snap_group[snap_group['displayName'] == 'football']
        if ball_data.empty:
            continue
        ball_x = ball_data['x'].values[0]
        ball_y = ball_data['y'].values[0]
        ball_pos = np.array([ball_x, ball_y])

        # Identify QB
        qb_candidates = snap_group.dropna(subset=['nflId'])
        qb_candidates = qb_candidates[qb_candidates['nflId'].isin(qb_positions)]
        if qb_candidates.empty:
            continue
        qb_y = qb_candidates['y'].values[0]
        qb_id = int(qb_candidates['nflId'].values[0])

        player_info_list = []
        for idx, row in snap_group.iterrows():
            if row['displayName'] == 'football':
                continue
            club = row['club']
            if club == pos_team:
                sign = 1.0
            elif club == def_team:
                sign = -1.0
            else:
                continue

            if pd.isna(row['nflId']):
                continue

            x, y = row['x'], row['y']
            s = row['s']
            o_deg = row['o']
            theta = angle_to_radians_for_influence(o_deg)

            try:
                area = gaussian_influence_area(x, y, theta, s, ball_pos)
            except Exception as e:
                print(f"Error computing gaussian_influence_area for nflId {row['nflId']} in {gamePlayId}: {e}")
                area = 0.0
            area *= sign

            nflId = int(row['nflId'])
            player_info = players_map.get(nflId, {})
            if not isinstance(player_info, dict):
                print(f"Unexpected player_info type for nflId {nflId}: {type(player_info)}")
                position = None
            else:
                position = player_info.get('position', None)

            player_info_list.append({
                'nflId': nflId,
                'x': x,
                'y': y,
                'theta': theta,
                'club': club,
                'position': position,
                'area': area,
                'speed': s
            })

        offensive_players = [p for p in player_info_list if p['club'] == pos_team]
        defensive_players_info = [p for p in player_info_list if p['club'] == def_team]

        # Compute top and bottom influence
        # top: y > qb_y, bottom: y < qb_y
        top_area_sum = 0.0
        bottom_area_sum = 0.0
        for p in player_info_list:
            if p['nflId'] == qb_id:
                continue
            if p['y'] > qb_y:
                top_area_sum += p['area']
            else:
                bottom_area_sum += p['area']

        # Determine primary (greater influence) and secondary side
        if top_area_sum >= bottom_area_sum:
            primary_side = 'top'
            secondary_side = 'bottom'
        else:
            primary_side = 'bottom'
            secondary_side = 'top'

        # Identify eligible receivers based on updated rules
        # Separate offensive players by side
        top_side_players = [p for p in offensive_players if p['y'] >= qb_y]
        bottom_side_players = [p for p in offensive_players if p['y'] < qb_y]

        # Sort players by y-coordinate for consistency
        top_side_sorted = sorted(top_side_players, key=lambda p: p['y'])
        bottom_side_sorted = sorted(bottom_side_players, key=lambda p: p['y'])

        # Determine eligible receivers on each side
        eligible_top = get_eligible_receivers_side(top_side_sorted, 'top', LOS_x)
        eligible_bottom = get_eligible_receivers_side(bottom_side_sorted, 'bottom', LOS_x)

        eligible_receivers_nflIds = eligible_top.union(eligible_bottom)

        # Compute pressures only for eligible receivers
        def compute_pressures_for_players(off_players, def_players, eligible_ids):
            '''Compute pressures for eligible offensive players.

            Args:
                off_players: list of offensive player dicts
                def_players: list of defensive player dicts
                eligible_ids: set of eligible receivers' nflIds

            Return:
                List of tuples (nflId, pressure)
            '''
            pressures = []
            for op in off_players:
                if op['nflId'] in eligible_ids and op['position'] in ['WR', 'TE', 'RB', 'FB']:
                    pressure_val = compute_player_pressure(op, def_players)
                    pressures.append((op['nflId'], pressure_val))
            return pressures

        off_pressures = compute_pressures_for_players(offensive_players, defensive_players_info, eligible_receivers_nflIds)

        # Separate pressures by side
        top_ids = {p['nflId'] for p in top_side_players}
        bottom_ids = {p['nflId'] for p in bottom_side_players}

        top_eligible_pressures = [(nid, val) for (nid, val) in off_pressures if nid in top_ids]
        bottom_eligible_pressures = [(nid, val) for (nid, val) in off_pressures if nid in bottom_ids]

        # Sort each side by pressure ascending
        top_eligible_pressures.sort(key=lambda x: x[1])
        bottom_eligible_pressures.sort(key=lambda x: x[1])

        # According to influence, primary_side first
        if primary_side == 'top':
            combined = top_eligible_pressures + bottom_eligible_pressures
        else:
            combined = bottom_eligible_pressures + top_eligible_pressures

        # Create list of dicts {nflId: pressure}
        final_pressures = [{str(nid): val} for (nid, val) in combined]
        final_results[gamePlayId] = final_pressures

    return final_results


def main():
    # Read plays and players data
    try:
        plays_df = pd.read_csv('plays.csv', dtype={
            'gameId': 'int32',
            'playId': 'int32',
            'possessionTeam': 'object',
            'defensiveTeam': 'object',
            'yardlineNumber': 'int32'
        })
    except Exception as e:
        print(f"Error reading plays.csv: {e}")
        return

    try:
        players_df = pd.read_csv('players.csv', dtype={'nflId': 'int32', 'position': 'object'})
    except Exception as e:
        print(f"Error reading players.csv: {e}")
        return

    # Create plays_map with necessary fields: possessionTeam, defensiveTeam, yardlineNumber
    plays_key = plays_df.apply(lambda row: f"{row['gameId']}_{row['playId']}", axis=1)
    plays_map = plays_df.set_index(plays_key)[['possessionTeam', 'defensiveTeam', 'yardlineNumber']].to_dict(orient='index')

    # Check for duplicate keys
    if len(plays_key) != len(plays_map):
        print("Warning: Duplicate gameId_playId detected. Only the last occurrence is kept.")

    # Create a set of QB nflIds for quick lookup
    qb_positions = set(players_df[players_df['position'] == 'QB']['nflId'])

    # Create players_map with position info
    players_map = players_df.set_index('nflId').to_dict(orient='index')

    # Find all tracking files
    tracking_pattern = 'tracking_week_*.csv'
    tracking_files = glob.glob(tracking_pattern)

    if not tracking_files:
        print(f"No files found for pattern '{tracking_pattern}'")
        return

    # Use multiprocessing Pool to process files in parallel
    num_processes = min(cpu_count(), len(tracking_files))
    pool = Pool(processes=num_processes)

    # Partial function with fixed arguments
    process_func = partial(process_tracking_file, plays_map=plays_map, players_map=players_map, qb_positions=qb_positions)

    try:
        results = pool.map(process_func, tracking_files)
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        pool.close()
        pool.join()
        return

    pool.close()
    pool.join()

    # Merge all partial results into a single dictionary
    final_dict = {}
    print('Be ready to merge into final_dict')
    for partial_dict in results:
        if not isinstance(partial_dict, dict):
            print(f"Warning: Partial result is not a dict: {partial_dict}")
            continue
        for k, v in partial_dict.items():
            if k in final_dict:
                print(f"Warning: Duplicate gamePlayId {k} detected. Overwriting previous entry.")
            final_dict[k] = v

    # Serialize the final dictionary using orjson for faster performance
    print('Be ready to export to json')
    try:
        with open('static_eval.json', 'wb') as f:
            f.write(orjson.dumps(final_dict, option=orjson.OPT_INDENT_2))
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return

    print("static_eval.json has been generated.")


if __name__ == "__main__":
    main()
