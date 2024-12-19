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

Generate a json file that ESTIMATE players fatigue in current play

Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import csv
import json
from collections import defaultdict, OrderedDict
import math
import os
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import orjson


def load_player_play_data(player_play_file):
    """
    Load player_play data.

    Args:
        player_play_file (str): Path to the player_play.csv file.

    Returns:
        defaultdict: A dictionary mapping (gameId, playId) to a list of (nflId, teamAbbr).
    """

    player_data = defaultdict(list)
    with open(player_play_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gameId = row['gameId']
            playId = row['playId']
            nflId = row['nflId']
            teamAbbr = row['teamAbbr']
            player_data[(gameId, playId)].append((nflId, teamAbbr))
    return player_data


def load_plays_data(plays_file):
    """
    Load plays data.

    Args:
        plays_file (str): Path to the plays.csv file.

    Returns:
        dict: A dictionary mapping (gameId, playId) to (possessionTeam, defensiveTeam).
    """

    plays_info = {}
    with open(plays_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gameId = row['gameId']
            playId = row['playId']
            possessionTeam = row['possessionTeam']
            defensiveTeam = row['defensiveTeam']
            plays_info[(gameId, playId)] = (possessionTeam, defensiveTeam)
    return plays_info


def calculate_continuity(player_data, plays_info):
    """
    Calculate the number of consecutive plays each player has participated in.

    Args:
        player_data (defaultdict): Dictionary mapping (gameId, playId) to list of (nflId, teamAbbr).
        plays_info (dict): Dictionary mapping (gameId, playId) to (possessionTeam, defensiveTeam).

    Returns:
        OrderedDict: A dictionary mapping "gameId_playId" to a dictionary of {nflId: continuity_count}.
    """

    game_plays = defaultdict(list)
    for (gameId, playId) in plays_info.keys():
        game_plays[gameId].append(playId)
    for gameId in game_plays:
        game_plays[gameId].sort(key=lambda x: int(x))

    final_result = OrderedDict()
    player_continuity = {}  # {(gameId, nflId): continuity_count}
    prev_possession_team = {}

    for gameId in game_plays:
        player_continuity.clear()
        prev_possession_team[gameId] = None

        for playId in game_plays[gameId]:
            possessionTeam, _ = plays_info[(gameId, playId)]
            players = player_data.get((gameId, playId), [])
            current_continuity = {}

            # Determine offense-defense switch
            if possessionTeam != prev_possession_team[gameId]:
                player_continuity.clear()

            for nflId, teamAbbr in players:
                key = (gameId, nflId)
                if key in player_continuity:
                    player_continuity[key] += 1
                else:
                    player_continuity[key] = 1

                current_continuity[nflId] = player_continuity[key]

            final_result[f"{gameId}_{playId}"] = current_continuity
            prev_possession_team[gameId] = possessionTeam

    return final_result


def process_tracking_file(tracking_file):
    """
    Process a single tracking_week_*.csv file.

    Args:
        tracking_file (str): Path to the tracking_week_*.csv file.

    Returns:
        dict: Partial tracking data from the file.
    """

    tracking_data = defaultdict(list)

    with open(tracking_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gameId = row['gameId']
            playId = row['playId']
            nflId = row['nflId']
            event = row['event']
            try:
                x = float(row['x'])
                y = float(row['y'])
                s = float(row['s'])
            except ValueError:
                # If x, y, s cannot be converted to float, skip the row
                continue
            tracking_data[(gameId, playId, nflId)].append((event, x, y, s))

    return tracking_data


def load_tracking_data(tracking_files):
    """
    Load multiple tracking_week_*.csv files and organize them into time series data.

    Args:
        tracking_files (list): List of paths to tracking_week_*.csv files.

    Returns:
        defaultdict: A dictionary mapping (gameId, playId, nflId) to a list of (event, x, y, s).
    """

    tracking_data = defaultdict(list)

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_tracking_file, tracking_files)
    for partial_data in results:
        for key, value in partial_data.items():
            tracking_data[key].extend(value)
    return tracking_data


def compute_play_stats_single(key_data):
    """
    Compute running distance and maximum speed for a single (gameId, playId, nflId).

    Args:
        key_data (tuple): Tuple containing (key, data_points).

    Returns:
        tuple: (key, {"distance": dist, "max_speed": max_s})
    """

    key, data_points = key_data
    # Find the index of the ball_snap event
    snap_index = next((i for i, (event, x, y, s) in enumerate(data_points) if event == "ball_snap"), None)
    if snap_index is None:
        # If there is no ball_snap event, set distance and speed to 0
        return (key, {"distance": 0.0, "max_speed": 0.0})

    # Calculate distance and max speed from ball_snap onwards
    dist = 0.0
    max_s = 0.0
    prev_x, prev_y = None, None
    for (event, x, y, s) in data_points[snap_index:]:
        if prev_x is not None and prev_y is not None:
            dx = x - prev_x
            dy = y - prev_y
            step_dist = math.hypot(dx, dy)  # Euclidean distance
            dist += step_dist
        if s > max_s:
            max_s = s
        prev_x, prev_y = x, y

    return (key, {"distance": dist, "max_speed": max_s})


def compute_play_stats(tracking_data):
    """
    Compute running distance and maximum speed for each (gameId, playId, nflId).

    Args:
        tracking_data (dict): Dictionary mapping (gameId, playId, nflId) to list of (event, x, y, s).

    Returns:
        dict: A dictionary mapping (gameId, playId, nflId) to {"distance": dist, "max_speed": max_s}.
    """

    play_stats = {}
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_play_stats_single, tracking_data.items())
    play_stats = dict(results)
    return play_stats


def build_player_play_index(continuity_dict):
    """
    Build an index to quickly find the previous N-1 consecutive plays for each player in a game.

    Args:
        continuity_dict (dict): Dictionary mapping "gameId_playId" to {nflId: continuity_count, ...}.

    Returns:
        defaultdict: A dictionary mapping (gameId, nflId) to a list of (playId, continuity_count) sorted by playId.
    """

    player_game_plays = defaultdict(list)
    # Decompose gameId_playId into (gameId, playId)
    # Group by gameId and sort playId
    game_buckets = defaultdict(list)
    for gp_str in continuity_dict.keys():
        gameId, playId = gp_str.split("_")
        game_buckets[gameId].append(playId)

    # Sort playIds for each gameId
    for gameId in game_buckets:
        game_buckets[gameId].sort(key=lambda x: int(x))

    # Reconstruct player_game_plays based on sorted playIds
    for gameId in game_buckets:
        for playId in game_buckets[gameId]:
            gp_str = f"{gameId}_{playId}"
            for nflId, cont in continuity_dict[gp_str].items():
                player_game_plays[(gameId, nflId)].append((playId, cont))

    return player_game_plays


def compute_fatigue_factors_single(gp_str, player_dict, player_game_plays, play_stats):
    """
    Compute V_fatigue for a single "gameId_playId".

    Args:
        gp_str (str): "gameId_playId" string.
        player_dict (dict): {nflId: continuity_count, ...} for the current play.
        player_game_plays (dict): Mapping from (gameId, nflId) to list of (playId, continuity_count).
        play_stats (dict): Mapping from (gameId, playId, nflId) to {"distance": dist, "max_speed": max_s}.

    Returns:
        tuple: (gp_str, {nflId: V_fatigue, ...})
    """

    gameId, playId = gp_str.split("_")
    current_results = {}
    for nflId, N in player_dict.items():
        if N == 1:
            # First appearance or first play after rest
            V_fatigue = 1.0
        else:
            # Find the N-1 consecutive plays for the player
            plays_list = player_game_plays.get((gameId, nflId), [])
            # plays_list is [(playId, continuity_count), ...] sorted
            # Current playId corresponds to N, so the previous N-1 plays are at idx-(N-1) to idx-1
            idx = next((i for i, (pid, c) in enumerate(plays_list) if pid == playId), None)
            if idx is None or idx - (N - 1) < 0:
                # playId not found or not enough previous plays
                V_fatigue = 1.0
            else:
                prev_plays = plays_list[idx-(N-1):idx]  # N-1 plays
                # Calculate average distance and average max speed
                distances = []
                max_speeds = []
                for (p_pid, p_cont) in prev_plays:
                    stat = play_stats.get((gameId, p_pid, nflId), {"distance": 0.0, "max_speed": 0.0})
                    distances.append(stat["distance"])
                    max_speeds.append(stat["max_speed"])
                if not distances:
                    V_fatigue = 1.0
                else:
                    Distance_avg = sum(distances) / len(distances)
                    V_max_avg = sum(max_speeds) / len(max_speeds)
                    # Calculate Fatigue
                    fatigue_coeff = 0.004
                    V_reachable = 9
                    D_strd = 30
                    Fatigue = fatigue_coeff * N * (V_max_avg / V_reachable) * (Distance_avg / D_strd)
                    V_fatigue = 1 - Fatigue
                    # Ensure V_fatigue is not below 0
                    V_fatigue = max(V_fatigue, 0.0)

        # Round V_fatigue to 4 decimal places
        V_fatigue = round(V_fatigue, 4)
        current_results[nflId] = V_fatigue

    return (gp_str, current_results)


def compute_fatigue_factors(continuity_dict, play_stats):
    """
    Calculate V_fatigue based on continuity information and play statistics.

    V_fatigue calculation:
    1. If N = 1 (first appearance or first play after rest): V_fatigue = 1, this guys fully charged!
    2. If N > 1:
        - Find the previous N-1 consecutive plays of the player
        - Calculate the average running distance Distance_avg of these N-1 plays
        - Calculate the average maximum speed V_max_avg of these N-1 plays
        - Fatigue = 0.004 * N * (V_max_avg / 9) * (Distance_avg / 30)
        - V_fatigue = 1 - Fatigue

    Args:
        continuity_dict (dict): Dictionary mapping "gameId_playId" to {nflId: continuity_count, ...}.
        play_stats (dict): Dictionary mapping (gameId, playId, nflId) to {"distance": dist, "max_speed": max_s}.

    Returns:
        OrderedDict: A dictionary mapping "gameId_playId" to {nflId: V_fatigue, ...}.
    """

    result = OrderedDict()
    player_game_plays = build_player_play_index(continuity_dict)

    # Partial function for multiprocessing
    # partial_func = partial(compute_fatigue_factors_single, player_game_plays=player_game_plays, play_stats=play_stats)

    # multiprocess
    with Pool(processes=cpu_count()) as pool:
        tasks = [(gp_str, player_dict, player_game_plays, play_stats) for gp_str, player_dict in continuity_dict.items()]
        results_pool = pool.starmap(compute_fatigue_factors_single, tasks)

    for gp_str, fatigue_dict in results_pool:
        result[gp_str] = fatigue_dict

    return result

def main():
    """
    Main function to orchestrate data loading, processing, and output generation.

    Steps:
    1. Load player_play and plays data.
    2. Calculate player continuity.
    3. Load tracking data from multiple tracking_week_*.csv files.
    4. Compute play statistics (distance and max speed).
    5. Calculate V_fatigue factors based on continuity and play stats.
    6. Output the results to a JSON file.
    """

    player_play_file = "player_play.csv"
    plays_file = "plays.csv"
    tracking_pattern = "tracking_week_*.csv"

    # Load data
    player_data = load_player_play_data(player_play_file)
    plays_info = load_plays_data(plays_file)

    # Calculate player continuity
    continuity_dict = calculate_continuity(player_data, plays_info)

    # Find all tracking_week_*.csv files
    tracking_files = glob.glob(tracking_pattern)
    if not tracking_files:
        print(f"No tracking files found matching the pattern '{tracking_pattern}'.")
        return

    # Load tracking data
    tracking_data = load_tracking_data(tracking_files)
    # Compute distance and max speed for each play
    play_stats = compute_play_stats(tracking_data)

    # Calculate V_fatigue based on continuity and play_stats
    fatigue_dict = compute_fatigue_factors(continuity_dict, play_stats)

    # Output to JSON file using orjson
    output_file = "velocity_fatigue.json"
    with open(output_file, "wb") as f:  # use orjson to avoid computer crash
        f.write(orjson.dumps(fatigue_dict, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))

    print("velocity_fatigue.json has been generated.")


if __name__ == "__main__":
    main()
# Overall time cost: 8 min to run, 