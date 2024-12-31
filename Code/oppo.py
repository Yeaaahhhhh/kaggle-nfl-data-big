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

This python file will calculate opportunities for each WR, TE, FB, RB in each play, they will take simulation_engine.py
as a basic file that simulate again and again, simulation time depends on personal computer memories and GPU count.
!!Highly suggest: adjust the simulation number or pixels in simulation_engine to get final results faster!!

Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''

import os
import json
import orjson
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm  # For progress bar
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import UnivariateSpline

# Import the simulation function from simulation_engine.py
from simulation_engine import run_wave_simulation_for_play


def load_snap_elapse_receiver(filepath):
    """
    Load the snap_elapse_receiver.json file.

    Args:
        filepath (str): Path to the snap_elapse_receiver.json file.

    Returns:
        dict: Parsed JSON content.
    """

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Loaded snap_elapse_receiver.json with {len(data)} entries.")
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {filepath} is not a valid JSON file.")
        sys.exit(1)


def load_csv(filepath):
    """
    Load a CSV file into a pandas DataFrame with proper data types.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data with correct data types
    """

    try:
        # Explicitly specify data types for numeric columns
        dtype_dict = {
            'gameId': str,
            'playId': str,
            'nflId': str,
            'x': float,
            'y': float,
            'o': float,
            # Add other columns as needed...
        }

        df = pd.read_csv(filepath, dtype=dtype_dict)

        # Ensure numeric columns are float type
        numeric_columns = ['x', 'y', 'o']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"Loaded {filepath} with {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def load_json_file(filepath):
    """
    Load a JSON file into a dictionary.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.
    """

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Loaded {filepath} with {len(data)} entries.")
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {filepath} is not a valid JSON file.")
        sys.exit(1)


def get_relevant_game_play_pairs(snap_data, tracking_df):
    """
    Extract valid gameId and playId pairs that exist in tracking_df.

    Args:
        snap_data (dict): Data from snap_elapse_receiver.json
        tracking_df (pd.DataFrame): Tracking data DataFrame

    Returns:
        list of tuples: List of (gameId, playId) tuples that exist in both datasets
    """

    valid_pairs = []
    tracking_pairs = set(zip(tracking_df['gameId'], tracking_df['playId']))
    for key in snap_data.keys():
        try:
            game_id, play_id = key.split('_')
            if (game_id, play_id) in tracking_pairs:
                valid_pairs.append((game_id, play_id))
            else:
                print(f"Skipping {key}: Not found in tracking_week_1.csv.")
        except ValueError:
            print(f"Invalid key format: {key}. Expected 'gameId_playId'. Skipping.")
    print(f"Total valid gameId_playId pairs to simulate: {len(valid_pairs)}")
    return valid_pairs


def filter_players(tracking_df, players_df, game_id, play_id):
    """
    Filter players for a specific game and play, including only WR, TE, RB, FB positions.

    Args:
        tracking_df (pd.DataFrame): Tracking data DataFrame
        players_df (pd.DataFrame): Players data DataFrame
        game_id (str): Game ID to filter
        play_id (str): Play ID to filter

    Returns:
        dict: Dictionary mapping nflId to position for relevant players
    """

    # Filter specific game and play
    filtered_tracking = tracking_df[
        (tracking_df['gameId'] == game_id) &
        (tracking_df['playId'] == play_id)
    ]

    # Merge player positions
    merged_df = pd.merge(filtered_tracking, players_df[['nflId', 'position']], on='nflId', how='left')

    # Filter positions
    relevant_positions = ["WR", "TE", "RB", "FB"]
    filtered_players = merged_df[merged_df['position'].isin(relevant_positions)]

    nflid_to_position = filtered_players.set_index('nflId')['position'].to_dict()
    print(f"Filtered to {len(nflid_to_position)} WR, TE, RB, FB players for gameId={game_id}, playId={play_id}.")
    return nflid_to_position


def compute_average_redness(simulation_results, relevant_nflids, num_simulations):
    """
    Compute the average redness across multiple simulations for each player.
    
    Args:
        simulation_results (list of dict): List of redness_result dictionaries from simulations
        relevant_nflids (set): Set of nflIds to consider
        num_simulations (int): Number of simulations performed
    
    Returns:
        dict: Dictionary mapping nflId to list of average redness values over time
    """

    # Initialize data structures for sum and count
    sum_redness = defaultdict(lambda: np.zeros(71))
    count_redness = defaultdict(int)

    # Accumulate redness values across simulations
    for sim_result in simulation_results:
        for nflId, redness_list in sim_result.items():
            if nflId in relevant_nflids:
                sum_redness[nflId] += np.array(redness_list, dtype=float)
                count_redness[nflId] += 1

    # Calculate averages and handle missing data
    average_redness = {}
    for nflId in relevant_nflids:
        if count_redness[nflId] == num_simulations:
            # Calculate average if all simulations are present
            average = sum_redness[nflId] / num_simulations
            average = np.round(average, 3)  # Round to 3 decimal places
            average_redness[nflId] = average.tolist()
        else:
            print(f"Warning: nflId {nflId} has {count_redness[nflId]} simulations instead of {num_simulations}. Assigning zeros.")
            average_redness[nflId] = [0.0] * 71  # Default if not enough data

    return average_redness


def compute_derivative(average_redness):
    """
    Compute the derivative of average redness over time for each player using smoothing.
    
    Args:
        average_redness (dict): Dictionary mapping nflId to list of average redness values
    
    Returns:
        dict: Dictionary mapping nflId to list of derivative values
    """
    derivative_dict = {}
    dt = 0.1  # Time step
    
    for nflId, redness_values in average_redness.items():
        t = np.linspace(0, 7.0, 71)  # Time points from t=0.0 to t=7.0
        y = np.array(redness_values, dtype=float)
        
        try:
            # Use UnivariateSpline for smoothing, s=1 indicates smoothing level
            spline = UnivariateSpline(t, y, s=1)
            dy = spline.derivative()(t)
            # Round to 3 decimal places
            dy = np.round(dy, 3)
            derivative_dict[nflId] = dy.tolist()
        except Exception as e:
            print(f"Error smoothing and computing derivative for nflId {nflId}: {e}")
            derivative_dict[nflId] = [0.0] * 71  # Default if error occurs

    return derivative_dict


def worker_simulate_plas(game_play_pairs, gpu_id, tracking_df, players_df, plays_df, velocity_fatigue_dict,
                         static_eval_dict):
    """
    Worker function to run simulations for multiple plays on a specific GPU.

    Args:
        game_play_pairs (list of tuples): List of (gameId, playId) tuples assigned to this worker
        gpu_id (int): GPU ID to assign to this worker
        tracking_df (pd.DataFrame): Tracking data DataFrame
        players_df (pd.DataFrame): Players data DataFrame
        plays_df (pd.DataFrame): Plays data DataFrame
        velocity_fatigue_dict (dict): Velocity fatigue dictionary
        static_eval_dict (dict): Static evaluation dictionary

    Returns:
        dict: Dictionary mapping gameId_playId to nflId derivative values
    """

    # Assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    local_opportunity = {}

    for game_id, play_id in game_play_pairs:
        key = f"{game_id}_{play_id}"
        print(f"GPU {gpu_id} - Simulating {key}...")

        # Get relevant players for current play
        nflid_to_position = filter_players(tracking_df, players_df, game_id, play_id)
        relevant_nflids = set(nflid_to_position.keys())

        if not relevant_nflids:
            print(f"GPU {gpu_id} - No relevant players for {key}. Skipping.")
            continue

        simulation_results = []

        for sim_num in range(1, 101):  # Run 100 simulations per play
            print(f"  GPU {gpu_id} - Simulation {sim_num}/100 for {key}...")
            sim_result = run_wave_simulation_for_play(
                game_id=game_id,
                play_id=play_id,
                tracking_df=tracking_df,
                players_df=players_df,
                plays_df=plays_df,
                velocity_fatigue_dict=velocity_fatigue_dict,
                static_eval_dict=static_eval_dict,
                field_xmax=120,
                field_ymax=53.3,
                sim_time=7.0
            )
            if key in sim_result:
                simulation_results.append(sim_result[key])
                print(f"    GPU {gpu_id} - Simulation {sim_num} completed for {key}.")
            else:
                print(f"    GPU {gpu_id} - Warning: No simulation result for {key} in simulation {sim_num}.")

        if len(simulation_results) != 100:
            print(
                f"GPU {gpu_id} - Warning: Expected 100 simulations, got {len(simulation_results)} for {key}. Assigning zeros.")
            # Assign zeros if not enough simulations
            average_redness = {nflId: [0.0] * 71 for nflId in relevant_nflids}
        else:
            # calculate average redness
            average_redness = compute_average_redness(simulation_results, relevant_nflids, 100)

        # calculate derivative
        derivative_redness = compute_derivative(average_redness)

        # only include relevant players
        local_opportunity[key] = {}
        for nflId, derivative_values in derivative_redness.items():
            local_opportunity[key][nflId] = derivative_values

    return local_opportunity


def main():
    """
    Main execution function for the OPPO analysis system.
    
    This function:
    1. Loads all required data files
    2. Processes play data across multiple GPUs
    3. Computes opportunity metrics
    4. Saves results to output file
    """

    # File paths
    snap_elapse_receiver_path = "snap_elapse_receiver.json"
    tracking_week1_path = "tracking_week_1.csv"            # you can modify here to use other tracking files
    players_path = "players.csv"
    plays_path = "plays.csv"
    velocity_fatigue_path = "velocity_fatigue.json"
    static_eval_path = "static_eval.json"
    opportunity_output_path = "opportunity.json"        # you can output to seperate json file then merge them together

    print("Starting OPPO simulation...")

    # Load all required data
    snap_data = load_snap_elapse_receiver(snap_elapse_receiver_path)
    tracking_df = load_csv(tracking_week1_path)
    players_df = load_csv(players_path)
    plays_df = load_csv(plays_path)
    velocity_fatigue_dict = load_json_file(velocity_fatigue_path)
    static_eval_dict = load_json_file(static_eval_path)

    # Get all valid gameId_playId pairs
    valid_pairs = get_relevant_game_play_pairs(snap_data, tracking_df)

    if not valid_pairs:
        print("No valid gameId_playId pairs to simulate. Exiting.")
        sys.exit(0)

    # Distribute valid_pairs across 6 GPUs, choose your own number, the more the faster
    num_gpus = 6
    chunks = [[] for _ in range(num_gpus)]
    for idx, pair in enumerate(valid_pairs):
        chunks[idx % num_gpus].append(pair)

    print(f"Distributing {len(valid_pairs)} plays across {num_gpus} GPUs.")

    # Create parameter list for each worker
    worker_args = []
    for gpu_id in range(num_gpus):
        worker_args.append((
            chunks[gpu_id],
            gpu_id,
            tracking_df,
            players_df,
            plays_df,
            velocity_fatigue_dict,
            static_eval_dict
        ))

    # Initialize opportunity dictionary
    opportunity = {}

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all workers
        futures = [
            executor.submit(worker_simulate_plas, *args)
            for args in worker_args
        ]
        # Use tqdm for progress tracking
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running Workers"):
            try:
                worker_result = future.result()
                for key, player_dict in worker_result.items():
                    opportunity[key] = player_dict
            except Exception as exc:
                print(f"Worker generated an exception: {exc}")

    # Convert to regular dictionary for serialization
    opportunity = dict(opportunity)

    # Save opportunity dictionary to JSON file, keeping 3 decimal places
    try:
        # Use orjson for efficient serialization
        with open(opportunity_output_path, 'wb') as f:
            f.write(orjson.dumps(opportunity, option=orjson.OPT_INDENT_2))
        print(f"\nSuccessfully saved opportunity data to {opportunity_output_path}.")
    except Exception as e:
        print(f"Error saving opportunity data: {e}")


if __name__ == "__main__":
    main()
