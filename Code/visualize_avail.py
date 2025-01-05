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

A visualization program that visualize availibility of eligible receivers in 3D style.


Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline


def read_data(players_csv_path, normalized_json_path, static_eval_json_path):
    """
    Read and parse the three input files.
    
    Args:
        players_csv_path (str): The path to players.csv file.
        normalized_json_path (str): The path to normalized.json file.
        static_eval_json_path (str): The path to static_eval.json file.
    
    Returns:
        dict: A dictionary mapping nflId -> displayName from players.csv
        dict: normalized_data from normalized.json
        dict: static_eval_data from static_eval.json
    """

    # players.csv
    players_df = pd.read_csv(players_csv_path, dtype=str)
    nflid_to_name = dict(zip(players_df['nflId'], players_df['displayName']))

    # normalized.json
    with open(normalized_json_path, 'r', encoding='utf-8') as f:
        normalized_data = json.load(f)

    # static_eval.json
    with open(static_eval_json_path, 'r', encoding='utf-8') as f:
        static_eval_data = json.load(f)

    return nflid_to_name, normalized_data, static_eval_data



def read_tracking_info(tracking_csv_path, game_id, play_id):
    """
    Read tracking_week_1.csv to retrieve each player's jerseyNumber
    for the specified gameId and playId.
    
    Args:
        tracking_csv_path (str): The path to tracking_week_1.csv.
        game_id (str): e.g. "2022091200"
        play_id (str): e.g. "467"
    
    Returns:
        dict: A dictionary mapping nflId -> jerseyNumber (string) for that play.
    """

    tracking_df = pd.read_csv(tracking_csv_path, dtype=str)
    subset = tracking_df[
        (tracking_df['gameId'] == game_id) &
        (tracking_df['playId'] == play_id)
    ]

    jersey_dict = {}
    for _, row in subset.iterrows():
        nflid = row.get('nflId', None)
        jersey_num = row.get('jerseyNumber', None)
        if nflid and jersey_num:
            jersey_dict[nflid] = jersey_num

    return jersey_dict



def short_name_transform(full_name):
    """
    Transform a name like "Abc Bcd" or "K.J. Hia" to "A.Bcd" or "K.Hia".
    
    Args:
        full_name (str): The original display name
    
    Returns:
        str: The shortened name form
    """
    parts = full_name.split()
    if len(parts) >= 2:
        first = parts[0]  # "Abc" / "K.J."
        last = parts[-1]  # "Bcd" / "Hia"

        # Remove '.' from first name, keep only first letter
        first_clean = first.replace(".", "")
        if len(first_clean) > 0:
            first_letter = first_clean[0]
            return f"{first_letter}.{last}"
        else:
            return full_name
    else:
        return full_name



def prepare_data(game_play_id, nflid_to_name, normalized_data, static_eval_data):
    """
    Prepare the data needed for plotting:
      1) Extract the 5 players' nflIds for the specified game_play_id.
      2) Build the availability time-series, including t=0 from static_eval.
    
    Args:
        game_play_id (str): e.g. "2022091200_467".
        nflid_to_name (dict): Mapping nflId -> displayName.
        normalized_data (dict): Data from normalized.json.
        static_eval_data (dict): Data from static_eval.json.
    
    Returns:
        list: A list of the 5 nflIds in this play.
        list: A list of the corresponding 5 displayNames (raw).
        dict: {nflId: [72 floats for availability over time]}.
    """

    if game_play_id not in normalized_data:
        raise ValueError(f"{game_play_id} not found in normalized_data.")

    play_normalized = normalized_data[game_play_id]  # {nflId: [0.3945, 0.4537, ...]}
    all_nflids = list(play_normalized.keys())
    first_five_nflids = all_nflids[:5]

    # Build {nflId: pressure}, default=0 if missing
    static_pressures = {}
    if game_play_id in static_eval_data:
        for d in static_eval_data[game_play_id]:
            for k, v in d.items():
                static_pressures[k] = float(v)

    # Construct availability time series
    nflid_to_availability = {}
    for nflid in first_five_nflids:
        pressure = static_pressures.get(nflid, 0.0)
        t0_avail = 1.0 - pressure
        dynamic_values = play_normalized[nflid]  # length=71
        full_vals = [t0_avail] + dynamic_values  # total=72
        nflid_to_availability[nflid] = full_vals

    display_names = []
    for nflid in first_five_nflids:
        name = nflid_to_name.get(nflid, f"Unknown_{nflid}")
        display_names.append(name)

    return first_five_nflids, display_names, nflid_to_availability



def create_animation(
    nflids,
    raw_display_names,
    nflid_to_availability,
    jersey_dict,
    output_gif_path
):
    """
    Create a 3D animation with:
      1) Camera dropping from top-down (elev=90) to front view (elev=5) over many frames for smoothness.
      2) The y-axis initially shows the player names (color-coded). 
         When camera fully arrives at the bottom, the y-axis names disappear, leaving only the legend.
      3) The legend names are "A.Bcd (22)" style, 
         reading jerseyNumber from jersey_dict and using short_name_transform for the display name.
      4) X-axis always 3 units wide. 
         - If current time t < 1.5, x-axis = [0, 3].
         - If t >= 1.5, x-axis = [t-1.5, t+1.5], clamp if t+1.5 > 7 => [4, 7].
      5) More total frames => smoother animation.

    Args:
        nflids (list): The 5 player nflIds.
        raw_display_names (list): The original display names.
        nflid_to_availability (dict): {nflId: [72 floats for availability]}.
        jersey_dict (dict): {nflId -> jerseyNumber}.
        output_gif_path (str): Where to save gif.

    Returns:
        None
    """

    # times: 0..7.0 by 0.1 => ~72 points
    times = np.arange(0, 7.0001, 0.1)
    num_time_points = len(times)  # typically 72

    # We'll define a camera transition of 50 frames for a smooth drop
    camera_transition_frames = 50

    # Lines start to grow in the last 10 frames of that camera transition
    # => frames 40..49 => line_index=0..9
    # Then after frame=49 => camera stable => finish line growth
    # We'll do 2 frames per time step for the remaining points => slower => more frames => smoother

    # So total line points are 72. We use 10 in frames[40..49].
    # Remain 62 points => each step=2 frames => 124 frames
    # total_frames = 50 + 124=174 frames
    transition_start_growth = camera_transition_frames - 25  # =40
    used_in_transition = 10  # line points used in transition
    remain_points = num_time_points - used_in_transition  # e.g. 62
    slow_growth_frames = remain_points * 2  # e.g. 124
    total_frames = camera_transition_frames + slow_growth_frames  # e.g. 174

    # Prepare lines color, y-level
    color_map = ["black", "red", "green", "pink", "blue"]
    y_levels = np.arange(1, len(nflids) + 1)

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')

    # Clean background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    ax.grid(False)

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.9)

    # Axis labels & range
    ax.set_xlabel("time after snap (s)", labelpad=10)
    ax.set_zlabel("Availability", labelpad=10)
    plt.title("Dynamic Receivers' Availability vs Time", pad=15)

    # initial x-lim
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(nflids) + 1)
    ax.set_yticks([])
    ax.set_zlim(0, 1)

    # Shorten names + get jersey => build legend label "A.Bcd (22)"
    short_labels = []
    for nflid, full_name in zip(nflids, raw_display_names):
        short_n = short_name_transform(full_name)
        jersey_num = jersey_dict.get(nflid, "?")
        label_str = f"{short_n} ({jersey_num})"
        short_labels.append(label_str)

    # Legend
    line_handles = []
    for i in range(len(nflids)):
        lh = plt.Line2D([0], [0], color=color_map[i], lw=3, label=short_labels[i])
        line_handles.append(lh)

    # Create two separate legends for two rows
    # First row (3 players)
    first_row = ax.legend(
        handles=line_handles[:3],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        borderaxespad=0,
        fontsize=9,
        handlelength=2,
        columnspacing=1
    )

    # Add first legend to figure
    ax.add_artist(first_row)

    # Second row (2 players)
    ax.legend(
        handles=line_handles[3:],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.92),  # Adjust this value to control vertical spacing between rows
        ncol=2,
        frameon=False,
        borderaxespad=0,
        fontsize=9,
        handlelength=2,
        columnspacing=1
    )

    # Create lines
    lines = []
    for i in range(len(nflids)):
        line, = ax.plot([], [], [], color=color_map[i], lw=1)
        lines.append(line)

    # ------------
    # Place y-axis text initially (like "Abc Bcd") and remove after camera stable
    # ------------
    texts_3d = []
    for i, name in enumerate(raw_display_names):
        txt = ax.text(
            -0.5,         # x
            y_levels[i],  # y
            0,            # z
            name,
            color=color_map[i],
            ha='right',
            va='center',
            fontsize=9
        )
        texts_3d.append(txt)

    # ------------
    # Helper: camera elev based on frame
    # frames 0..49 => linearly from 90->5
    # after that => 5
    # ------------
    def get_camera_elev(f):
        if f >= camera_transition_frames:
            return 5
        prog = f / (camera_transition_frames * 1.0)  # 0..1
        return 90 - 85 * prog  # 90->5

    # ------------
    # Helper: line_index based on frame
    # frames <40 => no growth
    # frames 40..49 => normal speed => line_index = f - 40 => 0..9
    # frames >=50 => slower => each step=2 frames
    #   frames=50 => line_index=10
    #   frames=51 => line_index=10
    #   frames=52 => line_index=11
    # ...
    # general => 10 + (f-50)//2
    # ------------
    def get_line_index(f):
        if f < transition_start_growth:  # 40
            return None
        if transition_start_growth <= f < camera_transition_frames:  # 40..49
            return f - transition_start_growth  # 0..9
        # f >=50 => slow growth
        idx = used_in_transition + (f - camera_transition_frames)//2
        return idx

    # ------------
    # We want a 3-wide x-lim
    # If t<1.5 => [0,3]
    # if t>=1.5 => [t-1.5, t+1.5], clamp if t+1.5>7 => [4,7]
    # ------------
    def dynamic_xlim(t):
        if t < 1.5:
            return 0, 2
        left = t - 0.1
        right = t + 0.1
        if right > 7:
            return 6, 7
        return left, right

    def init_func():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines + texts_3d

    def update_func(frame):
        # 1) Camera
        elev_now = get_camera_elev(frame)
        ax.view_init(elev=elev_now, azim=-90)

        # 2) line_index
        idx = get_line_index(frame)
        if idx is None or idx < 0:
            # lines not growing yet
            return lines + texts_3d
        if idx >= num_time_points:
            idx = num_time_points - 1

        current_t = times[idx]

        # 3) Update lines with smooth interpolation
        for i, nflid in enumerate(nflids):
            x_data = times[:idx + 1]
            y_data = np.full_like(x_data, y_levels[i], dtype=float)
            z_data = nflid_to_availability[nflid][:idx + 1]
            
            # Add smooth interpolation
            if len(x_data) > 3:  # Need at least 4 points for cubic spline
                # Create more dense points for smooth animation
                x_smooth = np.linspace(x_data[0], x_data[-1], num=100)
                cs = CubicSpline(x_data, z_data)
                z_smooth = cs(x_smooth)
                
                lines[i].set_data(x_smooth, np.full_like(x_smooth, y_levels[i]))
                lines[i].set_3d_properties(z_smooth)
            else:
                # Fall back to original data for first few points
                lines[i].set_data(x_data, y_data)
                lines[i].set_3d_properties(z_data)

        # 4) Move x-axis => 3 wide, sync from t=1.5
        xmin, xmax = dynamic_xlim(current_t)
        ax.set_xlim(xmin, xmax)

        # Update ticks, e.g. 0.5 interval
        tick_vals = np.arange(
            np.floor(xmin*2)/2,
            np.ceil(xmax*2)/2 + 0.01,
            1
        )
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([f"{v:.1f}" for v in tick_vals])

        # 5) Once camera stable => frame >= camera_transition_frames => remove y-axis text
        if frame >= camera_transition_frames:
            for txt_obj in texts_3d:
                txt_obj.set_visible(False)

        return lines + texts_3d

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update_func,
        frames=total_frames,
        init_func=init_func,
        blit=False,
        interval=10  # ms per frame => ~30 fps
    )

    ani.save(output_gif_path, writer='pillow', fps=3)
    print(f"Animation saved to {output_gif_path}")


def main():
    """
    Main function:
      1) Read data from players.csv, normalized.json, static_eval.json
      2) Also read jerseyNumber from tracking_week_1.csv
      3) Prepare data for a specific game_play_id
      4) Create animation with camera drop, y-axis name show/hide,
         legend with "A.Bcd (22)" format, and x-axis sync movement.
    """

    # File paths (adjust if needed)
    players_csv = "players.csv"
    normalized_json = "normalized.json"
    static_eval_json = "static_eval.json"
    tracking_csv = "tracking_week_1.csv"

    # Output
    output_gif = "rece_ava1.gif"

    # Target game & play
    game_id_str = "2022091200"
    play_id_str = "467"
    game_play_id = f"{game_id_str}_{play_id_str}"

    # 1) Read base data
    nflid_to_name, normalized_data, static_eval_data = read_data(
        players_csv,
        normalized_json,
        static_eval_json
    )

    # 2) Read tracking info to get jerseyNumber
    jersey_dict = read_tracking_info(
        tracking_csv_path=tracking_csv,
        game_id=game_id_str,
        play_id=play_id_str
    )

    # 3) Prepare data
    nflids, raw_display_names, nflid_to_availability = prepare_data(
        game_play_id,
        nflid_to_name,
        normalized_data,
        static_eval_data
    )

    # 4) Create animation
    create_animation(
        nflids,
        raw_display_names,
        nflid_to_availability,
        jersey_dict,
        output_gif
    )


if __name__ == "__main__":
    main()
