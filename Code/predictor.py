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

Prediction goes from this file, it will return overall top-1 accuracy and top-2 accuracy and each team accuracy


Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import json
import os
import csv


def load_json_file(file_path):
    """
    Load and parse a JSON file.
    
    Args:
        file_path (str): The path to the JSON file.
    
    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File {file_path} contains invalid JSON.")
        return {}


def parse_plays_csv(csv_path):
    """
    Parse the plays.csv file to map (gameId, playId) -> possessionTeam.
    
    Args:
        csv_path (str): The path to the plays.csv file.
    
    Returns:
        dict: A dictionary with keys as (gameId, playId) and 
              values as the corresponding possessionTeam.
    """
    game_play_to_team = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                game_id = row.get('gameId')
                play_id = row.get('playId')
                team = row.get('possessionTeam')
                if game_id and play_id and team:
                    game_play_to_team[(game_id, play_id)] = team
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
    return game_play_to_team


def calculate_frame_index(time_value):
    """
    Calculate the frame index based on the given time value.
    
    Args:
        time_value (float): The time value from snap_elapse_receiver.json.
    
    Returns:
        int: Calculated frame index.
    """
    return int(time_value * 10 + 1)


def extract_normalized_values(normalized_data, game_play_id, frame_index):
    """
    Extract the normalized values for each nflId at the specified frame index.
    
    Args:
        normalized_data (dict): The normalized.json data.
        game_play_id (str): The gameId_playId key.
        frame_index (int): The index to extract from each nflId's list.
    
    Returns:
        dict: A dictionary of nflId to its corresponding value at frame_index.
    """
    nfl_values = {}
    game_data = normalized_data.get(game_play_id, {})
    for nfl_id, values in game_data.items():
        if frame_index < len(values):
            nfl_values[nfl_id] = values[frame_index]
        else:
            # Handle index out of range by assigning a default value
            nfl_values[nfl_id] = 0.0
    return nfl_values


def get_static_order(static_eval_data, game_play_id):
    """
    Retrieve the static evaluation order for nflIds in a game_play_id.
    
    Args:
        static_eval_data (dict): The static_eval.json data.
        game_play_id (str): The gameId_playId key.
    
    Returns:
        list: A list of nflIds in the order they appear in static_eval.json.
    """
    static_order = []
    game_static = static_eval_data.get(game_play_id, [])
    for entry in game_static:
        for nfl_id in entry.keys():
            static_order.append(nfl_id)
    return static_order


def sort_nfl_ids(nfl_values, static_order):
    """
    Sort nflIds based on their normalized values and static evaluation order.
    
    Args:
        nfl_values (dict): A dictionary of nflId to its corresponding value.
        static_order (list): The static evaluation order of nflIds.
    
    Returns:
        list: A list of nflIds sorted based on the criteria.
    """
    # Initial sorting based on normalized values in descending order
    sorted_nfl = sorted(nfl_values.items(), key=lambda x: x[1], reverse=True)
    
    # Adjust sorting for nflIds with value differences less than 0.02
    i = 0
    while i < len(sorted_nfl) - 1:
        current_nfl, current_val = sorted_nfl[i]
        next_nfl, next_val = sorted_nfl[i + 1]
        
        if abs(current_val - next_val) < 0.02:
            try:
                current_index = static_order.index(current_nfl)
            except ValueError:
                current_index = len(static_order)
            try:
                next_index = static_order.index(next_nfl)
            except ValueError:
                next_index = len(static_order)
            
            if current_index > next_index:
                # Swap positions
                sorted_nfl[i], sorted_nfl[i + 1] = sorted_nfl[i + 1], sorted_nfl[i]
                # Move back one step to recheck previous pairs
                if i > 0:
                    i -= 1
                continue
        
        i += 1
    
    # Extract only nflIds in sorted order
    ranked_nfl_ids = [nfl_id for nfl_id, _ in sorted_nfl]
    return ranked_nfl_ids


def write_errors(error_file, top1_errors, top2_errors):
    """
    Write the prediction errors to the error.txt file.
    
    Args:
        error_file (str): The path to the error.txt file.
        top1_errors (list): A list of Top-1 error tuples.
        top2_errors (list): A list of Top-2 error tuples.
    
    Returns:
        None
    """
    with open(error_file, 'w') as file:
        file.write("Top-1 Errors:\n")
        for game_play_id, predicted, actual in top1_errors:
            file.write(f"{game_play_id}: Predicted NFL ID {predicted}, Actual NFL ID {actual}\n")
        
        file.write("\nTop-2 Errors:\n")
        for game_play_id, predicted_list, actual in top2_errors:
            file.write(f"{game_play_id}: Predicted NFL IDs {predicted_list}, Actual NFL ID {actual}\n")



def main():
    """
    Main function to execute the prediction accuracy calculation and 
    compute team-level accuracy based on Top-1 results.
    
    Args:
        None
    
    Returns:
        None
    """
    # File paths
    snap_file = 'snap_elapse_receiver.json'
    normalized_file = 'normalized.json'
    static_eval_file = 'static_eval.json'
    plays_csv_file = 'plays.csv'
    error_output_file = 'error.txt'
    
    # Load JSON data
    snap_data = load_json_file(snap_file)
    normalized_data = load_json_file(normalized_file)
    static_eval_data = load_json_file(static_eval_file)
    
    # Parse plays.csv to get possessionTeam info
    game_play_to_team = parse_plays_csv(plays_csv_file)
    
    # Initialize ranking dictionary and error lists
    rank = {}
    top1_errors = []
    top2_errors = []
    
    # Initialize accuracy counters
    total_predictions = 0
    correct_top1 = 0
    correct_top2 = 0
    
    # Dictionary to track team-level stats: {teamName: [correct_count, total_count]}
    team_stats = {}
    
    # Iterate through each gameId_playId in snap_data
    for game_play_id, time_info in snap_data.items():
        # Extract time and actual nflId
        if not isinstance(time_info, dict) or len(time_info) != 1:
            continue  # Skip invalid entries
        time_str, actual_nfl_id = next(iter(time_info.items()))
        
        try:
            time_value = float(time_str)
        except ValueError:
            continue  # Skip entries with invalid time values
        
        # Filter out entries with time >= 6.5
        if time_value > 6.5:
            continue
        
        # Calculate frameIndex and frame_idx
        frame_index = calculate_frame_index(time_value)
        frame_idx = frame_index - 1
        
        # Extract normalized values
        nfl_values = extract_normalized_values(normalized_data, game_play_id, frame_idx)
        if not nfl_values:
            continue  # Skip if no nflIds are found
        
        # Get static evaluation order
        static_order = get_static_order(static_eval_data, game_play_id)
        
        # Sort nflIds based on criteria
        ranked_nfl_ids = sort_nfl_ids(nfl_values, static_order)
        rank[game_play_id] = ranked_nfl_ids
        
        # Increment total predictions
        total_predictions += 1
        
        # Convert the actual nflId to string
        actual_nfl_id_str = str(actual_nfl_id)
        
        # Top-1 Accuracy
        if ranked_nfl_ids and ranked_nfl_ids[0] == actual_nfl_id_str:
            correct_top1 += 1
            top1_correct_flag = True
        else:
            top1_correct_flag = False
            predicted_top1 = ranked_nfl_ids[0] if ranked_nfl_ids else None
            top1_errors.append((game_play_id, predicted_top1, actual_nfl_id_str))
        
        # Top-2 Accuracy
        if actual_nfl_id_str in ranked_nfl_ids[:2]:
            correct_top2 += 1
        else:
            predicted_top2 = ranked_nfl_ids[:2] if len(ranked_nfl_ids) >= 2 else ranked_nfl_ids
            top2_errors.append((game_play_id, predicted_top2, actual_nfl_id_str))
        
        # ----- Team-level accuracy tracking -----
        # Extract gameId and playId from game_play_id, e.g. "2021090900_1234"
        parts = game_play_id.split('_')
        if len(parts) != 2:
            continue  # Skip if format is not gameId_playId
        game_id, play_id = parts
        
        # Find the team from game_play_to_team
        if (game_id, play_id) in game_play_to_team:
            team_name = game_play_to_team[(game_id, play_id)]
            if team_name not in team_stats:
                team_stats[team_name] = [0, 0]  # [correct_count, total_count]
            
            # We only consider top-1 correctness
            if top1_correct_flag:
                team_stats[team_name][0] += 1
            team_stats[team_name][1] += 1
    
    # Calculate overall accuracies
    top1_accuracy = (correct_top1 / total_predictions) if total_predictions > 0 else 0
    top2_accuracy = (correct_top2 / total_predictions) if total_predictions > 0 else 0
    
    # Write errors to error.txt
    write_errors(error_output_file, top1_errors, top2_errors)
    
    # Print overall accuracies
    print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
    print(f"Top-2 Accuracy: {top2_accuracy:.2%}")
    
    # ----- Compute team-level accuracy -----
    # team_stats[team_name] = [correct_count, total_count]
    team_accuracy_dict = {}
    for team, stats in team_stats.items():
        correct_count, total_count = stats
        if total_count > 0:
            acc = correct_count / total_count
        else:
            acc = 0
        team_accuracy_dict[team] = acc
    
    # Sort teams by accuracy in descending order and construct a final dict
    sorted_teams = sorted(team_accuracy_dict.items(), key=lambda x: x[1], reverse=True)
    team_accuracy_ranking = {team: acc for team, acc in sorted_teams}
    
    # Print team accuracy ranking
    print("Team accuracy ranking (Top-1):")
    print(team_accuracy_ranking)


if __name__ == "__main__":
    main()
