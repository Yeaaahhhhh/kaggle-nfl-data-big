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

This file will just simply show you the receive result, C means complete, F means not receive the ball

Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import json
import pandas as pd


def read_json_file(file_path):
    """
    Reads and parses a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary representation of the JSON file.
    """
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON.")
        return {}


def read_and_filter_csv(file_path, game_play_set):
    """
    Reads the CSV file and filters rows based on the provided gameId_playId set.

    Args:
        file_path (str): The path to the CSV file.
        game_play_set (set): A set of gameId_playId strings to filter the CSV.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only relevant rows.
    """
    try:
        plays_df = pd.read_csv(file_path, usecols=['gameId', 'playId', 'passResult', 'possessionTeam'])
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Error: The file {file_path} could not be parsed.")
        return pd.DataFrame()

    plays_df['gameId_playId'] = plays_df['gameId'].astype(str) + '_' + plays_df['playId'].astype(str)
    filtered_plays = plays_df[plays_df['gameId_playId'].isin(game_play_set)].copy()

    return filtered_plays


def build_receive_result(json_data, filtered_plays):
    """
    Constructs the receive_result dictionary based on JSON data and filtered CSV plays.

    Args:
        json_data (dict): The parsed JSON data from snap_elapse_receiver.json.
        filtered_plays (pd.DataFrame): The filtered plays DataFrame.

    Returns:
        dict: A dictionary with gameId_playId as keys and [possessionTeam, passResult] as values.
    """
    receive_result = {}

    # Create a mapping from gameId_playId to [possessionTeam, passResult]
    play_mapping = filtered_plays.set_index('gameId_playId').to_dict('index')

    for game_play_id in json_data.keys():
        if game_play_id in play_mapping:
            possession_team = play_mapping[game_play_id].get('possessionTeam', 'Unknown')
            pass_result = play_mapping[game_play_id].get('passResult')

            if pd.isna(pass_result) or pass_result == 'C':
                result = 'C'
            else:
                result = 'F'

            receive_result[game_play_id] = [possession_team, result]

    return receive_result


def main():
    """
    Main function to execute the processing of JSON and CSV files to build receive_result.
    """
    json_file_path = 'snap_elapse_receiver.json'
    csv_file_path = 'plays.csv'

    # Read JSON data
    snap_data = read_json_file(json_file_path)
    if not snap_data:
        return

    game_play_set = set(snap_data.keys())

    # Read and filter CSV data
    filtered_plays = read_and_filter_csv(csv_file_path, game_play_set)
    if filtered_plays.empty:
        print("No matching gameId_playId combinations found in plays.csv.")
        return

    receive_result = build_receive_result(snap_data, filtered_plays)

    print(receive_result)


if __name__ == "__main__":
    main()
