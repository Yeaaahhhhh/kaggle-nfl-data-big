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

Create a simulation engine for the game of NFL. The simulation engine is used to simulate the game.
This simulation engine is based on the wave model, and will be used in the future for calculation purpose.

Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


# simulation_engine.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Import required classes and methods from wave_model.py
from wave_model import (
    WaveSimulation,
    local_direction_angles,
    color_decay,
    get_vmax_dict     # Contains default v_max for each direction
)

# ========== 1) Read CSV/JSON files ==========

def read_data():
    """
    Reads required CSV/JSON files and returns corresponding DataFrames and dictionaries.
    Assumes all data files are in the same directory as simulation_engine.py.
    
    Args:
        None: Uses files from current directory
    
    Returns:
        tuple: Contains (tracking_df, players_df, plays_df, velocity_fatigue_dict, static_eval_dict)
        for use in simulation setup
    """
    tracking_df = pd.read_csv("tracking_week_1.csv", dtype={"gameId": str, "playId": str, "nflId": str})
    players_df  = pd.read_csv("players.csv",         dtype={"nflId": str})
    plays_df    = pd.read_csv("plays.csv",           dtype={"gameId": str, "playId": str})
    
    with open("velocity_fatigue.json", "r") as f:
        velocity_fatigue_dict = json.load(f)
    with open("static_eval.json", "r") as f:
        static_eval_dict = json.load(f)
    # If player_play.csv is needed, it can be read here:
    # player_play_df = pd.read_csv("player_play.csv", dtype={"gameId": str, "playId": str, "nflId": str})

    return tracking_df, players_df, plays_df, velocity_fatigue_dict, static_eval_dict


def get_ball_snap_info(tracking_df, game_id, play_id):
    """
    Retrieves all player records at ball_snap event for specified gameId and playId.
    Filters tracking data to get player positions at snap.
    
    Args:
        tracking_df (pd.DataFrame): Tracking data containing player positions
        game_id (str): Game identifier
        play_id (str): Play identifier
    
    Returns:
        pd.DataFrame: Contains [nflId, x, y, o, gameId, playId, event] for all players at snap
    """
    subset = tracking_df[
        (tracking_df["gameId"] == game_id) &
        (tracking_df["playId"] == play_id) &
        (tracking_df["event"] == "ball_snap")
    ].copy()
    return subset


def merge_position_info(ball_snap_df, players_df):
    """
    Merges ball snap data with player information to include position details.
    Combines positional data with player attributes.
    
    Args:
        ball_snap_df (pd.DataFrame): Ball snap event data
        players_df (pd.DataFrame): Player reference data with positions
    
    Returns:
        pd.DataFrame: Combined DataFrame with player positions and attributes
    """
    merged_df = pd.merge(
        ball_snap_df,
        players_df[["nflId", "position", "displayName"]],
        on="nflId",
        how="left"
    )
    return merged_df


def compute_player_vmax(position, fatigue_value):
    """
    Calculates maximum forward velocity based on player position and fatigue.
    Implements position-specific speed limits modified by fatigue factor.
    
    Args:
        position (str): Player's position (QB, RB, WR, etc.)
        fatigue_value (float): Fatigue multiplier between 0 and 1
    
    Returns:
        float: Maximum forward velocity adjusted for position and fatigue
    """

    base_forward_speed = 0.0
    
    # --- Offense ---
    if position == "QB":
        base_forward_speed = 8.0
    elif position in ["RB", "FB"]:
        base_forward_speed = 9.0
    elif position in ["C", "G", "T"]:
        base_forward_speed = 3.0
    elif position == "TE":
        base_forward_speed = 8.0
    elif position == "WR":
        base_forward_speed = 11.0
    
    # --- Defense ---
    elif position in ["NT", "DT"]:
        base_forward_speed = 3.0
    elif position == "DE":
        base_forward_speed = 4.0
    elif position == "ILB":
        base_forward_speed = 9.0
    elif position == "OLB":
        base_forward_speed = 6.0
    elif position in ["FS", "SS"]:
        base_forward_speed = 10.0
    elif position == "CB":
        base_forward_speed = 11.0
    
    return base_forward_speed * fatigue_value


def prepare_players_info(ball_snap_df, plays_df, velocity_fatigue_dict, static_eval_dict):
    """
    Processes player data to determine identities, speeds, and movement capabilities.
    Identifies offense/defense, applies fatigue, calculates max speeds, and determines forward eligibility.
    
    Args:
        ball_snap_df (pd.DataFrame): Player positions at snap
        plays_df (pd.DataFrame): Play context information
        velocity_fatigue_dict (dict): Player fatigue values
        static_eval_dict (dict): Player evaluation data
    
    Returns:
        pd.DataFrame: Enhanced player information with movement capabilities and context
    """

    if ball_snap_df.empty:
        return pd.DataFrame()
    
    game_id = ball_snap_df.iloc[0]["gameId"]
    play_id = ball_snap_df.iloc[0]["playId"]
    
    # Get possessionTeam, defensiveTeam etc. from plays_df
    row_play = plays_df[
        (plays_df["gameId"] == game_id) &
        (plays_df["playId"] == play_id)
    ]
    if row_play.empty:
        # Set default values if no match found
        possession_team = None
        defensive_team  = None
        yardline_number = None
        home_score      = 0
        visitor_score   = 0
        quarter         = 1
    else:
        row_play = row_play.iloc[0]
        possession_team = row_play["possessionTeam"]
        defensive_team  = row_play["defensiveTeam"]
        yardline_number = row_play["yardlineNumber"]
        home_score      = row_play["preSnapHomeScore"]
        visitor_score   = row_play["preSnapVisitorScore"]
        quarter         = row_play["quarter"]
    
    # Get playDirection from ball_snap_df
    play_direction = ball_snap_df.iloc[0]["playDirection"] if not ball_snap_df.empty else "right"
    
    # Build output DataFrame
    output_df = ball_snap_df.copy()
    side_list = []
    can_forward_list = []
    v_max_forward_list = []
    fatigue_value_list = []
    
    for idx, row in output_df.iterrows():
        nflId     = row["nflId"]
        position  = row["position"]
        # Determine offense/defense based on position
        if position in ["QB","RB","FB","WR","TE","C","G","T"]:
            side = "offense"
        else:
            side = "defense"
        side_list.append(side)
        
        # Check and correct player orientation
        current_angle = row["o"]
        
        # Handle 'NA' values
        if pd.isna(current_angle) or current_angle == 'NA':
            # Set default orientation based on side and play direction
            if play_direction == "left":
                current_angle = 270 if side == "offense" else 90
            else:  # play_direction == "right"
                current_angle = 90 if side == "offense" else 270
            output_df.at[idx, "o"] = current_angle
        else:
            # Convert to float if it's not already
            current_angle = float(current_angle)
            
            if play_direction == "left":
                # Offense should be between 180-359.9 degrees
                if side == "offense":
                    if not (180 <= current_angle <= 359.9):
                        output_df.at[idx, "o"] = 270  # Default value
                # Defense should be between 0-180 degrees
                else:
                    if not (0 <= current_angle <= 180):
                        output_df.at[idx, "o"] = 90   # Default value
            else:  # play_direction == "right"
                # Offense should be between 0-180 degrees
                if side == "offense":
                    if not (0 <= current_angle <= 180):
                        output_df.at[idx, "o"] = 90   # Default value
                # Defense should be between 180-359.9 degrees
                else:
                    if not (180 <= current_angle <= 359.9):
                        output_df.at[idx, "o"] = 270  # Default value
        
        # Calculate fatigue value
        combined_key = f"{game_id}_{play_id}"
        fatigue_dict_for_play = velocity_fatigue_dict.get(combined_key, {})
        fatigue_val = fatigue_dict_for_play.get(nflId, 1.0)
        fatigue_value_list.append(fatigue_val)
        
        # Maximum forward velocity
        base_fwd = compute_player_vmax(position, fatigue_val)
        v_max_forward_list.append(base_fwd)
        
        # Check if player can move forward
        if side=="offense":
            if position in ["RB", "FB", "WR", "TE"] and nflId in static_eval_dict:
                can_forward_list.append(True)
            else:
                can_forward_list.append(False)
        else:
            # Defense can always move forward by default
            can_forward_list.append(True)
    
    output_df["side"]           = side_list
    output_df["fatigue_value"]  = fatigue_value_list
    output_df["v_max_forward"]  = v_max_forward_list
    output_df["can_forward"]    = can_forward_list
    # Add information from plays_df for decision logic
    output_df["yardlineNumber"] = yardline_number
    output_df["home_score"]     = home_score
    output_df["visitor_score"]  = visitor_score
    output_df["quarter"]        = quarter
    output_df["playDirection"]  = play_direction
    
    return output_df


class MultiPlayerWaveSimulator:
    """
    A multi-player wave diffusion simulator that demonstrates:
    1) How WR/RB/FB offensive players determine their breakthrough directions (angles)
    2) How CBs match one-on-one with WR/RB and inherit their initial directions
    3) How other offensive/defensive positions use default decision logic
    
    Args:
        players_info_df (pd.DataFrame): Contains all players' info at ball_snap moment including x, y, o, 
            position, side, can_forward, v_max_forward, etc.
        xmax (float, optional): Field width. Defaults to 120
        ymax (float, optional): Field height. Defaults to 53.3
        nx (int, optional): Grid x resolution. Defaults to 1200
        ny (int, optional): Grid y resolution. Defaults to 530
        dt (float, optional): Time step. Defaults to 0.1
        t_end (float, optional): Total simulation time. Defaults to 7.0
    
    Returns:
        None: Initializes simulator state and player data structures
    """

    def __init__(self, 
                 players_info_df: pd.DataFrame,
                 xmax=120, ymax=53.3,
                 nx=120, ny=53,
                 dt=0.1, t_end=7.0):
        """
        players_info_df: Contains all players' x, y, o, position, side, can_forward, v_max_forward...
        xmax, ymax: Field dimensions
        nx, ny: Grid resolution
        dt: Time step
        t_end: Total simulation time
        """
        self.players_info_df = players_info_df.reset_index(drop=True)
        self.xmax = xmax
        self.ymax = ymax
        self.nx   = nx
        self.ny   = ny
        self.dt   = dt
        self.t_end = t_end
        self.n_frames = int(np.floor(self.t_end / self.dt))

        # Initialize grid
        self.xx = np.linspace(0, self.xmax, self.nx, endpoint=False)
        self.yy = np.linspace(0, self.ymax, self.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.xx, self.yy)

        # Color array: positive values = blue (defense), negative values = red (offense)
        self.color_array = np.zeros((self.ny, self.nx), dtype=float)
        offensive_positions = ["WR", "TE", "RB", "FB"]
        self.redness_result = {nflId: [] for nflId in self.players_info_df[
            self.players_info_df["position"].isin(offensive_positions)
        ]["nflId"].unique()}
        self.players_data = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # =============== (1) WR breakthrough points & angle random selection  ==================
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        wr_df = self.players_info_df[self.players_info_df["position"] == "WR"].copy()
        defenders_df = self.players_info_df[self.players_info_df["side"] == "defense"]
        
        # Pre-compute WR breakthrough points
        wr_breakthrough_points = {}
        for _, wr_row in wr_df.iterrows():
            wr_id = wr_row["nflId"]
            btp_points = self._compute_wr_breakthrough_points(
                wr_x=wr_row["x"],
                wr_y=wr_row["y"],
                play_dir=wr_row["playDirection"],
                defenders_df=defenders_df
            )
            wr_breakthrough_points[wr_id] = btp_points

        # Let each WR randomly select a breakthrough angle and store it
        self.wr_initial_angle = {}  # {wr_nflId : angle_in_degrees}
        for i_wr, row_wr in wr_df.iterrows():
            wr_id  = row_wr["nflId"]
            x_init = row_wr["x"]
            y_init = row_wr["y"]

            possible_pts = wr_breakthrough_points.get(wr_id, [])
            if possible_pts:
                target_point = random.choice(possible_pts)
                dx = target_point[0] - x_init
                dy = target_point[1] - y_init
                angle_deg = (np.degrees(np.arctan2(dx, dy))) % 360
            else:
                # Fallback direction if no breakthrough points (e.g., 0 degrees = up)
                angle_deg = 0.0

            self.wr_initial_angle[wr_id] = angle_deg

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # =============== (2) CB one-on-one match to a WR/RB and inherit their direction  =========
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Include RB/FB in the range that can be locked on by CBs
        wr_rb_df = self.players_info_df[
            self.players_info_df["position"].isin(["WR","RB","FB"])
        ].copy()

        # Leave an empty dictionary for RB/FB angle (to be assigned later)
        self.rb_fb_initial_angle = {}  # {rb_or_fb_nflId : angle_in_degrees}

        cb_df = self.players_info_df[self.players_info_df["position"] == "CB"].copy()
        claimed_wr_rb = set()
        cb_assignments = {}  # { cb_nflId : matched_wr_rb_nflId }

        # We'll inherit angles from WR/RB later in the loop when position=="CB"
        # For now, just do "shortest distance matching"

        for i_cb, cb_row in cb_df.iterrows():
            cb_nflId = cb_row["nflId"]
            cb_x     = cb_row["x"]
            cb_y     = cb_row["y"]

            # Calculate distance to each WR/RB
            wr_rb_df["dist_to_cb"] = np.sqrt(
                (wr_rb_df["x"] - cb_x)**2 +
                (wr_rb_df["y"] - cb_y)**2
            )
            wr_rb_df_sorted = wr_rb_df.sort_values("dist_to_cb")

            assigned = False
            for _, candidate in wr_rb_df_sorted.iterrows():
                candi_nflId = candidate["nflId"]
                if candi_nflId not in claimed_wr_rb:
                    cb_assignments[cb_nflId] = candi_nflId
                    claimed_wr_rb.add(candi_nflId)
                    assigned = True
                    break

            if not assigned:
                # If all WR/RB are occupied, leave as None
                cb_assignments[cb_nflId] = None

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # =============== (3) Let each RB/FB randomly select a CB/OLB direction ================
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        defenders_cb_olb_df = self.players_info_df[
            (self.players_info_df["side"] == "defense") &
            (self.players_info_df["position"].isin(["CB","OLB"]))
        ].copy()

        # Instead of calculating breakthrough points, let each RB/FB randomly select a defender (CB/OLB) direction
        # Later in the loop, we'll use self.rb_fb_initial_angle to get the angle

        rb_fb_df = self.players_info_df[
            self.players_info_df["position"].isin(["RB", "FB"])
        ].copy()

        for _, rb_fb_row in rb_fb_df.iterrows():
            rb_fb_id = rb_fb_row["nflId"]
            x_init   = rb_fb_row["x"]
            y_init   = rb_fb_row["y"]

            if not defenders_cb_olb_df.empty:
                chosen_def = defenders_cb_olb_df.sample(n=1).iloc[0]
                def_x = chosen_def["x"]
                def_y = chosen_def["y"]
                dx = def_x - x_init
                dy = def_y - y_init
                angle_deg = (np.degrees(np.arctan2(dx, dy))) % 360
            else:
                angle_deg = 0.0  # Fallback if no available defenders

            self.rb_fb_initial_angle[rb_fb_id] = angle_deg

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # =============== (4) Iterate through all players and construct self.players_data =============
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for idx, row in self.players_info_df.iterrows():
            nflId    = row["nflId"]
            position = row["position"]
            side     = row["side"]
            x_init   = row["x"]
            y_init   = row["y"]
            o_init   = row["o"]
            can_fwd  = row["can_forward"]
            v_fwd    = row["v_max_forward"]

            initial_decision = None
            target_info      = None

            # ===== WR: Use the custom direction we calculated =====
            if position == "WR":
                angle_deg = self.wr_initial_angle.get(nflId, 0.0)
                initial_decision = "change_face_direction"
                target_info = {
                    "angle": angle_deg,
                    "next_decision": "forward"
                }

            # ===== RB / FB: Use the random direction we calculated =====
            elif position in ["RB", "FB"]:
                angle_deg = self.rb_fb_initial_angle.get(nflId, 0.0)
                initial_decision = "change_face_direction"
                target_info = {
                    "angle": angle_deg,
                    "next_decision": "forward"
                }

            # ===== CB: If matched to a WR / RB, inherit their direction =====
            elif position == "CB":
                matched_wr_rb = cb_assignments.get(nflId, None)
                if matched_wr_rb is not None:
                    # Check matched_wr_rb's position
                    #    If WR => use wr_initial_angle
                    #    If RB/FB => use rb_fb_initial_angle
                    angle_deg = 90.0  # Fallback
                    matched_row = self.players_info_df[
                        self.players_info_df["nflId"] == matched_wr_rb
                    ]
                    if not matched_row.empty:
                        matched_pos = matched_row.iloc[0]["position"]
                        if matched_pos == "WR":
                            angle_deg = self.wr_initial_angle.get(matched_wr_rb, 90.0)
                        else:
                            angle_deg = self.rb_fb_initial_angle.get(matched_wr_rb, 90.0)
                    
                    initial_decision = "change_face_direction"
                    target_info = {
                        "angle": angle_deg,
                        "next_decision": "forward"
                    }
                else:
                    # If no match to WR/RB, use a default decision
                    initial_decision = "behind"

            # ===== Other positions: Fall back to your original default strategy =====
            else:
                initial_decision = self.get_initial_decision(position)

            # Fallback if still not set
            if initial_decision is None:
                initial_decision = self.get_initial_decision(position)

            # ========== Generate speed dictionary and scale ==========
            base_v_dict = get_vmax_dict(initial_decision)
            ratio = v_fwd / 11.0 if v_fwd > 0 else 0
            for k in base_v_dict:
                base_v_dict[k] *= ratio

            # ========== Construct p_data ==========
            p_data = {
                "nflId": nflId,
                "position": position,
                "side": side,
                "x": x_init,
                "y": y_init,
                "global_angle": o_init,
                "current_decision": initial_decision,
                "v_dict": base_v_dict,
                "can_forward": can_fwd,
                "wave_age": 0.0,
                "is_collided": False,
                "playDirection": row["playDirection"],
                "yardlineNumber": row["yardlineNumber"],
                "quarter": row["quarter"],
                "home_score": row["home_score"],
                "visitor_score": row["visitor_score"],
                # For WR, record the list of breakthrough points
                "breakthrough_points": (
                    wr_breakthrough_points.get(nflId, None)
                    if position == "WR" else None
                ),
                "target_info": target_info
            }
            self.players_data.append(p_data)


    def _calculate_current_wave_center(self, p_data):
        """
        Calculates the current wave front's center point for a given player.
        Uses exponential integration logic similar to wave_model._calculate_wave_center_at_time.
        
        Arg:
            p_data (dict): Player data containing current position, wave age, and velocity info
        
        Returns:
            tuple: (x,y) coordinates of the current wave front's center point
        """
        old_x = p_data["x"]
        old_y = p_data["y"]
        T     = p_data["wave_age"]
        
        # main_dir_key => e.g., "forward","left-forward"...
        main_dir_key = p_data["current_decision"]  
        v_max = p_data["v_dict"].get(main_dir_key, 0.0)

        if T <= 0 or v_max <= 0:
            return (old_x, old_y)

        tau = 1.3
        r_dir = v_max * (T - tau + tau*np.exp(-T/tau))
        if r_dir<0: 
            r_dir=0
        
        # Calculate global angle (degrees)
        global_angle_deg = self._get_global_angle(p_data["global_angle"], main_dir_key)
        rad = np.radians(global_angle_deg)
        
        x_new = old_x + r_dir * np.sin(rad)
        y_new = old_y + r_dir * np.cos(rad)
        return (x_new, y_new)


    def _get_main_direction_key(self, decision, base_angle):
        """
        Determines the 8-direction key based on decision and current base angle.
        Can return the decision itself if it matches one of the 8 directions.
        
        Args:
            decision (str): Player's current decision
            base_angle (float): Player's current base angle
        
        Returns:
            str: Direction key (e.g., "forward", "left", etc.) for wave propagation
        """

        # Simplified approach
        if decision in ["forward","behind","left","right",
                        "left-forward","right-forward","left-behind","right-behind",
                        "left-up","right-up"]:
            return decision
        else:
            
            return "forward"


    def _get_global_angle(self, base_angle, local_dir_key):
        """
        Converts local direction key and base angle to global angle coordinates.
        Maps player-relative directions to absolute angles in the field coordinate system.
        
        Args:
            base_angle (float): Player's base orientation angle
            local_dir_key (str): Local direction key (e.g., "forward", "left-up")
        
        Returns:
            float: Global angle in degrees (0-360) for wave propagation calculations
        """

        # This can also be implemented with a mapping, similar to wave_model:
        local_angle = {
            "forward": 0.0,
            "right-up": 45.0,
            "right": 90.0,
            "right-behind": 135.0,
            "behind": 180.0,
            "left-behind": 225.0,
            "left": 270.0,
            "left-up": 315.0
        }
        # If local_dir_key="left-forward", convert to local_angle=-45, etc
        # Demonstrating only a few cases here:
        if local_dir_key == "left-forward":
            offset = -45.0
        elif local_dir_key == "right-forward":
            offset = 45.0
        elif local_dir_key == "left-behind":
            offset = -135.0
        elif local_dir_key == "right-behind":
            offset = 135.0
        else:
            offset = local_angle.get(local_dir_key, 0.0)

        return (base_angle + offset) % 360
    

    def get_initial_decision(self, position):
        """
        Determines initial movement decision based on player position.
        Implements default strategic behavior for different football positions.
        
        Arg:
            position (str): Player's position (QB, RB, WR, etc.)
        
        Returns:
            str: Initial decision key (e.g., "forward", "behind", "left-forward") for wave simulation
        """

        # --- Offense ---
        if position == "QB":
            # Already set to behind at ball snap
            return "behind"
        
        elif position == "C":
            # Center C can be set to forward or behind
            return "behind"
        elif position == "G":
            # Guard G
            return "behind"  # Or forward, depending on need
        elif position == "T":
            # Tackle T, can differentiate left/right tackle
            return "left-behind"
        elif position == "TE":
            # TE often on the sides, can be set to left-forward or right-forward
            return "right-forward"
        elif position == "WR":
            # WR initially set to forward (task: change_face_direction then forward, simplified to direct forward)
            return "forward"

        # --- Defense ---
        if position in ["NT","DT"]:
            # Nose tackle/defensive tackle
            return "forward"
        elif position == "DE":
            # Can choose left-forward or right-forward based on position
            return "left-forward"
        elif position == "ILB":
            return random.choice(["left-behind", "right-behind"])
        elif position == "OLB":
            # Right-forward if on right side, left-forward if on left side
            return "right-forward"
        elif position in ["FS","SS"]:
            return random.choice(["left-behind", "right-behind"])
        elif position == "CB":
            # Modified: Randomly choose forward or behind
            return random.choice(["forward", "behind"])

        # Fallback for unknown positions
        return "forward"


    def run_simulation(self):
        """
        Execute the main simulation loop, updating player positions and calculating redness values.
        No visualization is performed.
        """
        for frame in range(self.n_frames + 1):
            t = frame * self.dt
            if t > self.t_end:
                break
            
            # Apply color decay
            self.color_array = color_decay(self.color_array)
            
            # Update all players
            self.update_all_players(self.dt)
            
            # Calculate redness
            self.calculate_redness()


    def update_all_players(self, dt):
        """
        Updates wave propagation state for all players in the simulation.
        Handles collisions, decision changes, and wave front calculations.
        
        Arg:
            dt (float): Time step for the update
        
        Returns:
            None: Updates internal player states and wave masks
        """

        wave_masks = []
        for p_data in self.players_data:
            # Handle WR delay
            if p_data.get("is_delayed", False):
                p_data["delay_timer"] -= dt
                if p_data["delay_timer"] <= 0:
                    p_data["is_delayed"] = False
                    # If there's a pending endzone forward
                    if p_data.get("pending_endzone_forward", False):
                        # Set new target direction based on playDirection
                        if p_data["playDirection"] == "left":
                            target_x = 0
                        else:  # right
                            target_x = 120
                        
                        # Calculate angle to endzone
                        dx = target_x - p_data["x"]
                        dy = p_data["yardlineNumber"] - p_data["y"]  # Use current y coordinate
                        target_angle = (np.degrees(np.arctan2(dx, dy))) % 360
                        
                        # Set change_face_direction
                        p_data["current_decision"] = "change_face_direction"
                        p_data["target_info"] = {
                            "angle": target_angle,
                            "next_decision": "forward"
                        }
                        p_data["wave_age"] = 0.0
                        p_data["pending_endzone_forward"] = False

            # Handle decision changes
            if p_data["current_decision"] == "change_face_direction" and p_data["target_info"]:
                # Execute turn
                p_data["global_angle"] = p_data["target_info"]["angle"]
                
                # Get next decision (different for CB and WR)
                next_decision = p_data["target_info"].get("next_decision", "forward")
                
                # Switch to next decision
                p_data["current_decision"] = next_decision
                
                # Reset wave age
                p_data["wave_age"] = 0.0
                
                # Update speed dictionary
                new_v_dict = get_vmax_dict(next_decision)
                ratio = p_data["v_dict"].get("forward", 11.0) / 11.0
                for k in new_v_dict:
                    new_v_dict[k] *= ratio
                p_data["v_dict"] = new_v_dict

            # Get wave mask
            mask = self.get_player_wave_mask(p_data, dt)
            wave_masks.append(mask)
        
        # Check collisions
        self.check_collisions(wave_masks)
        
        # Update color array
        for idx, (p_data, wave_mask) in enumerate(zip(self.players_data, wave_masks)):
            if p_data["side"] == "offense":
                self.color_array[wave_mask] = -1.0
            else:
                self.color_array[wave_mask] = 1.0
            
            # If collision flag, reset wave age
            if p_data.get("is_collided", False):
                p_data["wave_age"] = 0.0
                p_data["is_collided"] = False


    def get_player_wave_mask(self, p_data, dt):
        """
        Optimized wave mask calculation using vectorized operations and caching.
        """
        # Handle delayed state
        if p_data.get("is_delayed", False) and p_data.get("last_wave_center", None):
            old_x, old_y = p_data["x"], p_data["y"]
            old_wave_age = p_data["wave_age"]
            
            p_data["x"], p_data["y"] = p_data["last_wave_center"]
            p_data["wave_age"] = p_data["last_wave_age"]
            
            mask = self._calculate_wave_mask(p_data)
            
            p_data["x"], p_data["y"] = old_x, old_y
            p_data["wave_age"] = old_wave_age
            
            return mask

        if not p_data.get("is_delayed", False):
            p_data["wave_age"] += dt
        
        return self._calculate_wave_mask(p_data)


    def _calculate_wave_mask(self, p_data):
        """
        Optimized wave mask calculation using:
        1. Vectorized operations
        2. Pre-computed values
        3. Efficient boolean operations
        4. Reduced memory allocations
        """
        # Early exit for zero wave age
        if p_data["wave_age"] <= 0:
            return np.zeros((self.ny, self.nx), dtype=bool)

        # Calculate propagation radii once
        tau = 1.3
        T = p_data["wave_age"]
        exp_term = tau * np.exp(-T/tau)
        time_factor = T - tau + exp_term

        # Pre-calculate radii for all directions
        r_dict = {
            dir_key: max(0, base_v * time_factor)
            for dir_key, base_v in p_data["v_dict"].items()
        }

        # Use cached position differences if available
        px, py = p_data["x"], p_data["y"]
        cache_key = (px, py)
        
        if not hasattr(self, '_pos_cache'):
            self._pos_cache = {}
            self._pos_cache_size = 0
            self._max_cache_size = 1000  # Adjust based on memory constraints
        
        if cache_key in self._pos_cache:
            dx, dy, dist = self._pos_cache[cache_key]
        else:
            # Compute and cache position differences
            dx = self.X - px
            dy = self.Y - py
            dist = np.hypot(dx, dy)  # More efficient than sqrt(dx*dx + dy*dy)
            
            if self._pos_cache_size < self._max_cache_size:
                self._pos_cache[cache_key] = (dx, dy, dist)
                self._pos_cache_size += 1

        # Calculate angles once
        raw_angle = np.degrees(np.arctan2(dx, dy)) % 360
        base_angle = p_data["global_angle"]
        half_angle = 22.5
        wave_thickness = 0.5

        # Pre-allocate result mask
        wave_mask = np.zeros((self.ny, self.nx), dtype=bool)

        # Vectorized direction processing
        for dir_key, local_angle in local_direction_angles.items():
            r_dir = r_dict[dir_key]
            if r_dir <= 0:
                continue

            global_ang = (base_angle + local_angle) % 360
            lower_deg = (global_ang - half_angle) % 360
            upper_deg = (global_ang + half_angle) % 360
            
            # Distance mask
            lower_r = max(0, r_dir - wave_thickness)
            mask_dist = (dist >= lower_r) & (dist <= r_dir)

            # Angle mask - optimize based on angle range
            if lower_deg > upper_deg:
                mask_angle = (raw_angle >= lower_deg) | (raw_angle <= upper_deg)
            else:
                mask_angle = (raw_angle >= lower_deg) & (raw_angle <= upper_deg)

            # Update result mask using efficient boolean operations
            wave_mask |= (mask_dist & mask_angle)

        return wave_mask
    

    def _compute_wr_breakthrough_points(self, wr_x, wr_y, play_dir, defenders_df):
        """
        Generates and returns a list of potential breakthrough points for a WR based on defensive positions.
        Calculates midpoints between defenders and edge positions for route planning.
        
        Args:
            wr_x (float): WR's x coordinate
            wr_y (float): WR's y coordinate
            play_dir (str): Play direction ("left" or "right")
            defenders_df (pd.DataFrame): Defensive players' info containing x, y, position
        
        Returns:
            list: List of (x,y) tuples representing potential breakthrough points for the WR's route
        """

        points = []
        
        # 1) Get defensive player coordinates (positions in FS, SS, OLB, ILB, CB)
        # defenders_df is already filtered, further filter here
        df = defenders_df[ defenders_df["position"].isin(["FS","SS"]) ].copy()
        if df.empty:
            return points  # No defensive players, empty
        
        coords = df[["x","y"]].values  # shape=(N,2)
        
        # 2) Take midpoints between each pair
        n = coords.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                mx = 0.5*(coords[i,0]+coords[j,0])
                my = 0.5*(coords[i,1]+coords[j,1])
                points.append((mx,my))
        
        # 3) Find the farthest defensive player
        #   Example: (20,50) and (20,5) => closer to (18,49) is (20,50) => take (20,50) itself
        #   Then take midpoint with y=53.3 => e.g. ((20+20)/2, (50+53.3)/2)= (20,51.65)
        #   Demonstrating only finding y_max & y_min below
        df_sorted_by_y = df.sort_values("y")
        top_def = df_sorted_by_y.iloc[-1]  # y_max
        bot_def = df_sorted_by_y.iloc[0]   # y_min
        
        # Check which defensive player is closer to WR
        dist_top = np.hypot(wr_x - top_def["x"], wr_y - top_def["y"])
        dist_bot = np.hypot(wr_x - bot_def["x"], wr_y - bot_def["y"])
        if dist_top < dist_bot:
            # WR closer to top
            points.append((top_def["x"], top_def["y"]))
            # Midpoint with y=53.3 (or 53.0)
            my = 0.5*(top_def["y"] + self.ymax)
            points.append((top_def["x"], my))
        else:
            points.append((bot_def["x"], bot_def["y"]))
            # Midpoint with y=0
            my = 0.5*(bot_def["y"] + 0.0)
            points.append((bot_def["x"], my))
        
        # 4) Find the "last" defensive player (x_min or x_max)
        #    If play_dir=="left", last= x_min
        #    If play_dir=="right", last= x_max
        if play_dir=="left":
            df_sorted_by_x = df.sort_values("x")
            last_def = df_sorted_by_x.iloc[0]  # x_min
        else:
            df_sorted_by_x = df.sort_values("x")
            last_def = df_sorted_by_x.iloc[-1] # x_max
        
        # Midpoint with y=0 / y=53.3
        points.append((last_def["x"], last_def["y"]))
        points.append(( last_def["x"], 0.5*(last_def["y"] + 0.0) ))
        points.append(( last_def["x"], 0.5*(last_def["y"] + self.ymax) ))
        
        # 5) Find the nearest 4 defensive players (limited to FS,SS
        df_specific = df[df["position"].isin(["FS","SS"])].copy()
        if not df_specific.empty:
            df_specific["dist_wr"] = np.hypot(df_specific["x"]-wr_x, df_specific["y"]-wr_y)
            df_specific.sort_values("dist_wr", inplace=True)
            near4 = df_specific.head(4)
            for _, row in near4.iterrows():
                points.append((row["x"], row["y"]))
        
        # Deduplicate (optional)
        unique_points = list(set(points))
        return unique_points
    

    def _update_wr_decision_on_collision(self, wr_data):
        """
        Handles WR collision events by updating position and decision state.
        1) Moves WR to current arc endpoint
        2) Resets wave age
        3) Randomly selects new direction with delay
        4) Sets up endzone targeting after delay
        
        Arg:
            wr_data (dict): WR player data containing position and state information
        
        Returns:
            None: Updates WR position and decision state
        """

        # 1) Move to current arc endpoint
        x_new, y_new = self._calculate_current_wave_center(wr_data)
        wr_data["x"] = x_new
        wr_data["y"] = y_new
        # 2) Reset wave age
        wr_data["wave_age"] = 0.0
        
        # 3) Random decision + delay
        new_decision = np.random.choice(["forward","left-forward","right-forward","left","right"])
        wr_data["current_decision"] = new_decision

        # forward=11 => ratio
        old_forward_speed = wr_data["v_dict"].get("forward", 11.0)
        new_v = get_vmax_dict(new_decision)
        ratio = old_forward_speed / 11.0
        for k in new_v:
            new_v[k] *= ratio
        wr_data["v_dict"] = new_v

        wr_data["delay_timer"] = np.random.uniform(0.1,0.4)

        # 4) Check delay_timer in "update_all_players" to see if it's ended, then => endzone forward
        #    This is just a marker here
        wr_data["pending_endzone_forward"] = True  # New field, switch after delay ends


    def check_collisions(self, wave_masks):
        """
        Detects and handles collisions between players' wave fronts, implementing position-specific collision responses.
        Manages special collision cases like QB-DE interactions and OL-DL engagements.
        
        Args:
            wave_masks (list): List of boolean arrays representing each player's wave front coverage area,
                where each mask has shape (ny, nx) matching the simulation grid
        
        Returns:
            None: Updates player states and decisions based on detected collisions
        """

        n = len(self.players_data)
        def is_OL(pos): 
            return (pos in ["C","G","T"])
        def is_DL(pos):
            return (pos in ["NT","DT","DE"])
        
        # Handle other collision scenarios
        for i in range(n):
            for j in range(i+1, n):
                collide_mask = wave_masks[i] & wave_masks[j]
                if not np.any(collide_mask):
                    continue
                
                pi = self.players_data[i]
                pj = self.players_data[j]
                
                # If either player is WR, skip collision handling
                if pi["position"] == "WR" or pj["position"] == "WR":
                    continue

                # Handle QB & DE/OLB collision
                if (pi["position"] == "QB" and pj["position"] in ["DE","OLB"]):
                    # First, calculate current wave center for pi
                    x_new, y_new = self._calculate_current_wave_center(pi)
                    pi["x"], pi["y"] = x_new, y_new
                    pi["wave_age"] = 0.0  # New decision propagation starts from scratch

                    # Then switch
                    pi["current_decision"] = "right"  # Example
                    new_v = get_vmax_dict("right")
                    ratio = (pi["v_dict"].get("forward", 8.0)) / 11.0
                    for k in new_v:
                        new_v[k] *= ratio
                    pi["v_dict"] = new_v

                # If pj is QB
                elif (pj["position"] == "QB" and pi["position"] in ["DE","OLB"]):
                    x_new, y_new = self._calculate_current_wave_center(pj)
                    pj["x"], pj["y"] = x_new, y_new
                    pj["wave_age"] = 0.0

                    pj["current_decision"] = "left"
                    new_v = get_vmax_dict("left")
                    ratio = (pj["v_dict"].get("forward",8.0)) / 11.0
                    for k in new_v:
                        new_v[k] *= ratio
                    pj["v_dict"] = new_v

                # Defense (DL) & offensive OL collision
                if is_OL(pi["position"]) and is_DL(pj["position"]):
                    # Defense player pj decision -> left-forward or right-forward
                    pj["current_decision"] = np.random.choice(["left-forward","right-forward"])
                    new_v = get_vmax_dict(pj["current_decision"])
                    def_fwd = pj["v_dict"].get("forward",3.0)
                    ratio = def_fwd / 11.0 if def_fwd>0 else 0.27
                    for k in new_v:
                        new_v[k] *= ratio
                    pj["v_dict"] = new_v

                elif is_OL(pj["position"]) and is_DL(pi["position"]):
                    pi["current_decision"] = np.random.choice(["left-forward","right-forward"])
                    new_v = get_vmax_dict(pi["current_decision"])
                    def_fwd = pi["v_dict"].get("forward",3.0)
                    ratio = def_fwd / 11.0 if def_fwd>0 else 0.27
                    for k in new_v:
                        new_v[k] *= ratio
                    pi["v_dict"] = new_v


                # ============= 3) Defense (DL) & offensive OL (C/G/T) collision =============
                #   => Defense triggers left-forward or right-forward, then back to forward
                if is_OL(pi["position"]) and is_DL(pj["position"]):
                    # Defense player pj decision -> left-forward or right-forward
                    pj["current_decision"] = np.random.choice(["left-forward","right-forward"])
                    new_v = get_vmax_dict(pj["current_decision"])
                    # forward=3/4(e.g. NT/DT=3, DE=4) => ratio
                    def_fwd = pj["v_dict"].get("forward",3.0)
                    ratio = def_fwd / 11.0 if def_fwd>0 else 0.27
                    for k in new_v:
                        new_v[k] *= ratio
                    pj["v_dict"] = new_v

                elif is_OL(pj["position"]) and is_DL(pi["position"]):
                    pi["current_decision"] = np.random.choice(["left-forward","right-forward"])
                    new_v = get_vmax_dict(pi["current_decision"])
                    def_fwd = pi["v_dict"].get("forward",3.0)
                    ratio = def_fwd / 11.0 if def_fwd>0 else 0.27
                    for k in new_v:
                        new_v[k] *= ratio
                    pi["v_dict"] = new_v

                # ============= 4) RB/FB & DT collision => RB switches to forward/left-forward/right-forward =============
                # if (pi["position"] in ["RB","FB"] and pj["position"] in ["DT","NT"]):
                #     # Offensive player (RB/FB) decision
                #     pi["current_decision"] = np.random.choice(["forward","left-forward","right-forward"])
                #     new_v = get_vmax_dict(pi["current_decision"])
                #     # forward=9 => ratio
                #     ratio = (pi["v_dict"].get("forward",9.0)) / 11.0
                #     for k in new_v:
                #         new_v[k] *= ratio
                #     pi["v_dict"] = new_v

                # elif (pj["position"] in ["RB","FB"] and pi["position"] in ["DT","NT"]):
                #     pj["current_decision"] = np.random.choice(["forward","left-forward","right-forward"])
                #     new_v = get_vmax_dict(pj["current_decision"])
                #     ratio = (pj["v_dict"].get("forward",9.0)) / 11.0
                #     for k in new_v:
                #         new_v[k] *= ratio
                #     pj["v_dict"] = new_v

                # ============= (More rules can be added) =============

                # Note: In the demonstration, collisions are handled in one frame. If you want to "handle by nearest distance first", you can first collect collision pairs, sort them, and then execute the logic.

    def calculate_redness(self):
        """
        Optimized redness calculation using vectorized operations
        """
        # Convert player data to numpy arrays for faster computation
        offensive_players = [p for p in self.players_data if p["side"] == "offense"]
        defensive_players = [p for p in self.players_data if p["side"] == "defense"]

        # Pre-allocate arrays
        offensive_masks = {}
        defensive_mask = np.zeros((self.ny, self.nx), dtype=bool)
        
        # Vectorized wave mask calculation
        for p in defensive_players:
            defensive_mask |= self.get_player_wave_mask(p, self.dt)
        
        # Vectorized offensive calculations
        for p in offensive_players:
            nflId = p["nflId"]
            if nflId not in self.redness_result:
                self.redness_result[nflId] = []
            
            # Calculate player mask once
            player_mask = self.get_player_wave_mask(p, self.dt)
            offensive_masks[nflId] = player_mask
            
        # Vectorized exclusive mask calculation
        for nflId, player_mask in offensive_masks.items():
            other_offensive_mask = np.zeros((self.ny, self.nx), dtype=bool)
            for other_id, mask in offensive_masks.items():
                if other_id != nflId:
                    other_offensive_mask |= mask
            
            exclusive_mask = player_mask & (~other_offensive_mask)
            overlapping_defense = exclusive_mask & defensive_mask
            
            # Use numpy sum for faster calculation
            redness = np.sum(exclusive_mask) - np.sum(overlapping_defense)
            self.redness_result[nflId].append(float(redness))

def run_wave_simulation_for_play(
    game_id: str,
    play_id: str,
    tracking_df: pd.DataFrame,
    players_df: pd.DataFrame,
    plays_df: pd.DataFrame,
    velocity_fatigue_dict: dict,
    static_eval_dict: dict,
    field_xmax=120, 
    field_ymax=53.3,
    sim_time=7.0
):
    """
    Execute wave simulation for a single play and return redness values.
    
    Args:
        game_id (str): Unique identifier for the game
        play_id (str): Unique identifier for the play
        tracking_df (pd.DataFrame): Player tracking data
        players_df (pd.DataFrame): Player reference data
        plays_df (pd.DataFrame): Play-level information
        velocity_fatigue_dict (dict): Player fatigue values
        static_eval_dict (dict): Player evaluation data
        field_xmax (float, optional): Field width in yards
        field_ymax (float, optional): Field height in yards
        sim_time (float, optional): Total simulation duration
    
    Returns:
        dict: Dictionary containing redness values for each player
    """
    # 1) Get player positions at ball snap
    ball_snap_df = get_ball_snap_info(tracking_df, game_id, play_id)
    if ball_snap_df.empty:
        print(f"No ball_snap rows for gameId={game_id}, playId={play_id}.")
        return {}
    
    # 2) Merge position information
    merged_df = merge_position_info(ball_snap_df, players_df)
    
    # 3) Process player data
    processed_df = prepare_players_info(merged_df, plays_df, velocity_fatigue_dict, static_eval_dict)
    if processed_df.empty:
        print("No valid player data in this play.")
        return {}
    
    # 4) Create and run simulator (without visualization)
    sim = MultiPlayerWaveSimulator(
        processed_df,
        xmax=field_xmax, 
        ymax=field_ymax,
        nx=120, 
        ny=53,
        dt=0.1, 
        t_end=sim_time
    )
    
    # Run simulation without animation
    for frame in range(sim.n_frames + 1):
        t = frame * sim.dt
        if t > sim.t_end:
            break
            
        # Apply color decay
        sim.color_array = color_decay(sim.color_array)
        
        # Update all players
        sim.update_all_players(sim.dt)
        
        # Calculate redness
        sim.calculate_redness()
    

    #sim.print_offensive_redness()
    #print(f"{game_id}_{play_id}" + str(sim.redness_result))
    return {
        f"{game_id}_{play_id}": sim.redness_result
    }

    # Print redness results
    
    
    # Optionally save results to JSON file
    # with open("redness_result.json", "w") as f:
    #     json.dump(result, f, indent=4)


def main():
    # Read all data
    tracking_df, players_df, plays_df, velocity_fatigue_dict, static_eval_dict = read_data()

    # Specify gameId=2022091200, playId=467 (task requirement)
    game_id = "2022091200"
    play_id = "467"

    run_wave_simulation_for_play(
        game_id, play_id,
        tracking_df, players_df, plays_df,
        velocity_fatigue_dict, static_eval_dict,
        field_xmax=120, field_ymax=53.3,
        sim_time=7.0
    )

if __name__ == "__main__":
    main()
