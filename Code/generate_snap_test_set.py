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


import pandas as pd
import glob
import orjson
from multiprocessing import Pool, cpu_count
from functools import partial
import os


def process_one_tracking_file(tracking_file, player_play_df):
    """
    Process a single tracking_week file to extract the snap_time_elapse in game with the corresponding nflId information.

    Args.
        tracking_file (str): path to tracking_week file
        player_play_df (pd.DataFrame): full data of player_play.csv

    Return:
        dict: { "gameId_playId": { snap_time_elapse: nflId } }
    """
    usecols = ['gameId','playId','frameId','nflId','event']
    dtype_map = {
        'gameId': 'int32',
        'playId': 'int32',
        'frameId': 'int32',
        'nflId': 'float32',
        'event': 'object'
    }
    frame_gap = 0.1
    try:
        df = pd.read_csv(tracking_file, usecols=usecols, dtype=dtype_map)
    except Exception as e:
        print(f"Error reading {tracking_file}: {e}")
        return {}

    # group by gameId, playId
    final_dict = {}

    for (gameId, playId), group in df.groupby(['gameId','playId']):
        # See if there are pass_forward or handoff events in this round
        target_events = group[group['event'].isin(['pass_forward', 'handoff'])]
        ball_snap = group[group['event'] == 'ball_snap']

        if ball_snap.empty or target_events.empty:
            # No pass_forward or handoff events, or no ball_snap to calculate snap_time_elapse
            continue

        # Assuming that there can only be one snap corresponding to the event computation in a play
        snap_frame = ball_snap['frameId'].values[0]
        # Select the first pass_forward or handoff event
        target_frame = target_events['frameId'].iloc[0]

        snap_time_elapse = frame_gap * (target_frame - snap_frame)
        # keep 1 decimal
        snap_time_elapse = round(snap_time_elapse, 1)

        # Find the corresponding line in player_play.csv
        ppdf = player_play_df[(player_play_df['gameId']==gameId) & (player_play_df['playId']==playId)]

        if ppdf.empty:
            # No data found in player_play, skip.
            continue

        # Judging by wasTargettedReceiver
        receiver_info = ppdf[ppdf['wasTargettedReceiver'] == 1]
        if receiver_info.empty:
            # If there are no receivers marked as wasTargettedReceiver, look at hadRushAttempt
            rush_info = ppdf[ppdf['hadRushAttempt'] == 1]
            if rush_info.empty:
                # No players marked as RUSH and no target receivers
                continue
            else:
                # nflId using rush_info
                nflId = rush_info['nflId'].iloc[0]
        else:
            # use receiver's nflId
            nflId = receiver_info['nflId'].iloc[0]

        # write in dict
        key = f"{gameId}_{playId}"
        if key not in final_dict:
            final_dict[key] = {}
        final_dict[key][str(snap_time_elapse)] = int(nflId) if not pd.isna(nflId) else None

    return final_dict


def merge_dicts(dict_list):
    """
    Merge multiple dictionaries into one.

    Args.
        dict_list (list): list of dictionaries

    Return.
        dict: The merged dictionary
    """
    
    merged = {}
    for d in dict_list:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k in merged:
                # If the same key is repeated, it can be overwritten or merged depending on the requirements
                # Simple override here
                merged[k].update(v)
            else:
                merged[k] = v
    return merged


def main():

    # Assuming that the player_play.csv and tracking_week_*.csv files are in the current directory
    player_play_path = 'player_play.csv'
    tracking_pattern = 'tracking_week_*.csv'

    try:
        player_play_df = pd.read_csv(player_play_path, dtype={
            'gameId':'int32','playId':'int32','nflId':'int32','wasTargettedReceiver':'int8','hadRushAttempt':'int8'})
    except Exception as e:
        print(f"Error reading player_play.csv: {e}")
        return

    tracking_files = glob.glob(tracking_pattern)
    if not tracking_files:
        print(f"No files found for pattern '{tracking_pattern}'")
        return
    
    # use multiprocess
    num_processes = min(cpu_count(), len(tracking_files))
    pool = Pool(processes=num_processes)

    process_func = partial(process_one_tracking_file, player_play_df=player_play_df)

    try:
        results = pool.map(process_func, tracking_files)
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        pool.close()
        pool.join()
        return

    pool.close()
    pool.join()

    final_dict = merge_dicts(results)

    # Output json file using orjson
    # Require neat formatting with line breaks and indentation
    
    json_bytes = orjson.dumps(final_dict, option=orjson.OPT_INDENT_2)

    output_path = os.path.join(os.getcwd(), 'snap_elapse_receiver.json')
    with open(output_path, 'wb') as f:
        f.write(json_bytes)

    print("snap_elapse_receiver.json has been generated.")


if __name__ == "__main__":
    main()
