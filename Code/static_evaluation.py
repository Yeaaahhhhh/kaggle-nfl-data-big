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

Generate a text file that contains initial evaluation for QB pass priority:
Throw the ball to the left side? right side?
To whom? RB? WR? TE? FB?

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


def process_tracking_file(file_path, plays_map, players_map, qb_positions):
    """
    Process a single tracking_week file to compute pressures and identify eligible receivers.

    Steps:
    - Load tracking data, filter ball_snap
    - For each play:
      - Identify offense/defense teams, ball position, QB, etc.
      - Compute influence area for all players
      - Determine top_area_sum and bottom_area_sum (influence) to decide which side has higher influence for offensive
      - Based on playDirection:
        If direction='right', check LOS_x-1 <= x < LOS_x to check eligible receivers (WR,TE,RB,FB)
        If direction='left', check LOS_x < x <= LOS_x+1 check eligible receivers
      - Identify eligible receivers using outermost logic, covered up players not eligible
      - Compute pressures only for eligible WR,TE,RB,FB
      - Sort by pressure ascending within each side
      - The side with larger influence sum (area) goes first in the final list, then the other side
      - Final output format: [{"nflId": pressure}, ...]

    Args:
        file_path: prep to get all tracking_week_*.csv file
        plays_map: for each play, find out possesion team and defensive team
        players_map: same
        qb_position: locate quarter back position, his y will divide into to
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

        # Based on direction, define LOS eligibility range
        # If direction='right', check [LOS_x-1, LOS_x) for WR,TE,RB
        # If direction='left', check (LOS_x, LOS_x+1] for WR,TE,RB
        if play_dir == 'right':
            def is_in_range(x_coord):
                return (LOS_x - 1) <= x_coord < LOS_x
        else:  # play_dir == 'left'
            def is_in_range(x_coord):
                return LOS_x < x_coord <= (LOS_x + 1)

        # Separate by side for WR,TE,RB
        top_side_players = [p for p in offensive_players if p['y'] >= qb_y and p['position'] in ['WR','TE','RB','FB']]
        bottom_side_players = [p for p in offensive_players if p['y'] < qb_y and p['position'] in ['WR','TE','RB','FB']]

        # Filter players within LOS range based on play direction
        in_range_top = [p for p in top_side_players if is_in_range(p['x'])]
        in_range_bottom = [p for p in bottom_side_players if is_in_range(p['x'])]

        def compute_pressures_for_players(off_players, def_players):
            '''Compute pressures for offensive players.

            Args:
                off_players: list of all running style players
                def_players: list of all nearby def players

            Return:
                pressure list matched with nfl ID

            '''
            pressures = []
            for op in off_players:
                if op['position'] in ['WR', 'TE', 'RB','FB']:
                    pressure_val = compute_player_pressure(op, def_players)
                    pressures.append((op['nflId'], pressure_val))
            return pressures

        # If no WR, TE, RB in LOS range on either side, no eligibility filtering
        if not in_range_top and not in_range_bottom:
            # Just compute pressures normally
            off_pressures = compute_pressures_for_players(offensive_players, defensive_players_info)

            # separate by side
            top_ids = {p['nflId'] for p in top_side_players}
            bottom_ids = {p['nflId'] for p in bottom_side_players}

            # sort each side by pressure ascending
            top_side_pressures = [(nid, val) for (nid, val) in off_pressures if nid in top_ids]
            bottom_side_pressures = [(nid, val) for (nid, val) in off_pressures if nid in bottom_ids]

            top_side_pressures.sort(key=lambda x: x[1])
            bottom_side_pressures.sort(key=lambda x: x[1])

            # According to influence, primary_side first
            if primary_side == 'top':
                combined = top_side_pressures + bottom_side_pressures
            else:
                combined = bottom_side_pressures + top_side_pressures

            # Create list of dicts {nflId: pressure}
            final_pressures = [{str(nid): val} for (nid, val) in combined]
            final_results[gamePlayId] = final_pressures
            continue

        # If we have in-range players, determine eligibility
        def get_eligible_receivers_side(players_in_range_sorted, side='top'):

            ''' 
            This function will find out on each side who are the eligible receivers based on the rules

            Args:
                players_in_range_sorted: for those who are far far away from LOS (line of scrimmage)
                side: top, on the top side of QB
            
            Return:
                eligible: a list that contains all eligible receivers.

            '''
            eligible = []
            if not players_in_range_sorted:
                return eligible
            if side == 'top':
                outermost = players_in_range_sorted[-1]
            else:
                outermost = players_in_range_sorted[0]

            eligible.append(outermost['nflId'])
            outer_diff = abs(outermost['x'] - LOS_x)
            for player in players_in_range_sorted:
                if player['nflId'] == outermost['nflId']:
                    continue
                player_diff = abs(player['x'] - LOS_x)
                if player_diff < outer_diff:
                    # Closer to LOS_x => not eligible
                    continue
                else:
                    eligible.append(player['nflId'])
            return eligible

        # Sort players by y coordinate to find out each side players
        in_range_top_sorted = sorted(in_range_top, key=lambda p: p['y'])
        in_range_bottom_sorted = sorted(in_range_bottom, key=lambda p: p['y'])

        eligible_top = get_eligible_receivers_side(in_range_top_sorted, side='top')
        eligible_bottom = get_eligible_receivers_side(in_range_bottom_sorted, side='bottom')

        eligible_receivers_nflIds = set(eligible_top + eligible_bottom)

        # Compute pressures for all WR, TE, RB, FB
        off_pressures = compute_pressures_for_players(offensive_players, defensive_players_info)

        # Separate pressures by side and filter eligible
        top_ids = {p['nflId'] for p in top_side_players}
        bottom_ids = {p['nflId'] for p in bottom_side_players}

        top_eligible_pressures = [(nid, val) for (nid, val) in off_pressures if nid in top_ids and nid in eligible_receivers_nflIds]
        bottom_eligible_pressures = [(nid, val) for (nid, val) in off_pressures if nid in bottom_ids and nid in eligible_receivers_nflIds]

        # Sort each side by pressure ascending
        top_eligible_pressures.sort(key=lambda x: x[1])
        bottom_eligible_pressures.sort(key=lambda x: x[1])

        # Combine according to which side has bigger influence (primary_side)
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
    plays_df = pd.read_csv('plays.csv', dtype={
        'gameId': 'int32',
        'playId': 'int32',
        'possessionTeam': 'object',
        'defensiveTeam': 'object',
        'yardlineNumber': 'int32'
    })
    players_df = pd.read_csv('players.csv', dtype={'nflId': 'int32', 'position': 'object'})

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
    for partial_dict in results:
        if not isinstance(partial_dict, dict):
            print(f"Warning: Partial result is not a dict: {partial_dict}")
            continue
        for k, v in partial_dict.items():
            if k in final_dict:
                print(f"Warning: Duplicate gamePlayId {k} detected. Overwriting previous entry.")
            final_dict[k] = v

    # Serialize the final dictionary using orjson for faster performance
    try:
        with open('static_eval.json', 'wb') as f:
            f.write(orjson.dumps(final_dict, option=orjson.OPT_INDENT_2))
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return

    print("static_eval.json has been generated.")


if __name__ == "__main__":
    main()
