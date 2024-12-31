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

wave_model.py

Create a wave model for the game of NFL. The wave model is used to simulate the wave propagation of the game.
This is the core algorithm for future multiple-agent simulation.

Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation


##############################################################################
# 1) Decision System: Given a decision, returns maximum velocity v_max for 8 directions
##############################################################################
def get_vmax_dict(decision):
    """
    Converts a player's movement decision into maximum velocities for eight directions.
    Maps strategic decisions to velocity distributions that reflect realistic player movement capabilities.
    
    Arg:
        decision (str): Player's movement decision (e.g., "forward", "left", "right-behind", etc.)
    
    Returns:
        dict: Maximum velocities for 8 directions, where keys are direction names and values are
        corresponding max speeds. Used to calculate wave propagation speeds in different directions.
    """

    if decision == "forward":
        # Forward decision: use same velocity distribution
        return {
            "forward": 11,
            "left-up": 2,   
            "right-up": 2,
            "left": 1,
            "right": 1,
            "left-behind": 0.1,
            "behind": 0.1,
            "right-behind": 0.1
        }
    
    elif decision == "left-forward":
        return {
            "forward": 1,
            "left-up": 6,
            "right-up": 0.1,
            "left": 1.5,
            "right": 0.1,
            "left-behind": 0.1,
            "behind": 0.1,
            "right-behind": 0.1
        }
    
    elif decision == "right-forward":
        return {
            "forward": 1,
            "right-up": 6,
            "left-up": 0.1,
            "right": 2,
            "left": 0.1,
            "right-behind": 0.1,
            "behind": 0.1,
            "left-behind": 0.1
        }
    
    elif decision == "left":
        return {
            "forward": 0.5,
            "right-up": 0.5,
            "right": 0.1,
            "right-behind": 0.1,
            "behind": 0.1,
            "left-behind": 1,
            "left": 4,
            "left-up": 1.2
        }
    
    elif decision == "right":
        return {
            "forward": 0.5,
            "right-up": 1.5,
            "right": 4,
            "right-behind": 0.5,
            "behind": 0.1,
            "left-behind": 0.5,
            "left": 0.5,
            "left-up": 1
        }
    
    elif decision == "behind":
        return {
            "behind": 2.5,
            "left-behind": 1,
            "right-behind": 1,
            "left": 0.5,
            "right": 0.5,
            "left-up": 0.1,
            "right-up": 0.1,
            "forward": 0.1
        }
    
    elif decision == "left-behind":
        return {
            "behind": 1,
            "left-behind": 4,
            "right-behind": 0.5,
            "left": 1,
            "right": 0.5,
            "left-up": 0.5,
            "right-up": 0.5,
            "forward": 0.1
        }
    
    elif decision == "right-behind":
        return {
            "behind": 0.5,
            "left-behind": 0.5,
            "right-behind": 4,
            "left": 0.5,
            "right": 1,
            "left-up": 0.5,
            "right-up": 0.5,
            "forward": 0.1
        }
    
    elif decision == "change_face_direction":
        # No need to return velocity dictionary
        return {}
    
    else:
        # Use forward velocity distribution by default
        return {
            "forward": 11,
            "left-up": 2,   
            "right-up": 2,
            "left": 1,
            "right": 1,
            "left-behind": 0.1,
            "behind": 0.1,
            "right-behind": 0.1
        }

##############################################################################
# 2) Direction Table: Eight directions in "player local coordinate system" 
#    (clockwise, 0°=forward=player facing direction)
#    When converting to global coordinates, add player's base_angle


local_direction_angles = {
    "forward": 0.0,
    "right-up": 45.0,
    "right": 90.0,
    "right-behind": 135.0,
    "behind": 180.0,
    "left-behind": 225.0,
    "left": 270.0,
    "left-up": 315.0
}
half_angle = 22.5  # Half-angle of each sector


def color_decay(old_color):
    """
    Applies a decay factor to the current color value to simulate wave dissipation over time.
    
    Arg:
        old_color (float): Current color intensity value between 0 and 1
    
    Returns:
        color_decayed (float): Decayed color value, used to update the wave intensity visualization in the next frame
    """

    return old_color * 0.9


##############################################################################
# 4) WaveSimulation Main Class
##############################################################################

class VelocityTransitionManager:
    """
    Manages smooth transitions between different velocity states for realistic player movement.
    Handles acceleration and deceleration based on current speed and target speed.
    
    Args:
        None: Initialized with default values for velocity tracking
    
    Note: Individual method documentation follows below
    """

    def __init__(self):
        self.current_velocity = 0
        self.target_velocity = 0
        self.transition_start_time = 0
        self.transition_duration = 0
        self.is_transitioning = False
        

    def start_transition(self, current_v, target_v, current_time):
        """
        Initiates a velocity transition with appropriate duration based on current speed.
        
        Args:
            current_v (float): Current velocity value
            target_v (float): Target velocity to transition to
            current_time (float): Current simulation time
        
        Returns:
            bool: True if transition started successfully, False if cancelled due to high velocity
        """

        self.current_velocity = current_v
        self.target_velocity = target_v
        self.transition_start_time = current_time
        
        # Determine transition time based on current velocity
        if current_v <= 4:
            self.transition_duration = 0.1
        elif 4 < current_v <= 6:
            self.transition_duration = 0.3
        elif 6 < current_v <= 8:
            self.transition_duration = 1.0
        else:
            # Cancel transition if velocity is too high
            return False
            
        self.is_transitioning = True
        return True
        

    def get_current_velocity(self, current_time):
        """
        Calculates the current velocity during a transition using linear interpolation.
        
        Arg:
            current_time (float): Current simulation time
        
        Returns:
            float: Current interpolated velocity value used for wave propagation calculations
        """

        if not self.is_transitioning:
            return self.current_velocity
            
        dt = current_time - self.transition_start_time
        if dt >= self.transition_duration:
            self.is_transitioning = False
            return self.target_velocity
            
        # Calculate current velocity using linear interpolation
        progress = dt / self.transition_duration
        return self.current_velocity + (self.target_velocity - self.current_velocity) * progress


class WaveSimulation:

    def __init__(self,
                 xmax=120, ymax=53,
                 nx=1200, ny=530,
                 dt=0.1,
                 t_end=10.0,
                 initial_player_angle=270.0,
                 start_x=15, start_y=20):
        """
        xmax, ymax: Field dimensions
        nx, ny: Grid resolution (adjustable)
        dt: Time step
        t_end: Simulation end time
        initial_player_angle: Player's orientation in global coordinates (problem: 270°)
        start_x, start_y: Player's initial position
        """

        self.xmax, self.ymax = xmax, ymax
        self.nx, self.ny = nx, ny
        self.dt = dt
        self.t_end = t_end
        self.n_frames = int(np.floor(t_end / dt))
        self.initial_angle = initial_player_angle
        self.base_angle = initial_player_angle  # Current facing angle
        self.player_x = start_x
        self.player_y = start_y
        self.velocity_manager = VelocityTransitionManager()
        self.current_decision = None
        self.current_velocity = 0
        
        # Modify decision sequence to support parameterized decisions
        # Decision format: {"time": float, "decision": str, "angle": float (optional)}
        self.decision_timeline = [
            {"time": 0.0, "decision": "behind"},
            {"time": 3.0, "decision": "change_face_direction", "angle": 70.0},
            {"time": 3.0, "decision": "forward"}
        ]
        
        # Initialize grid and color arrays
        self._initialize_grid()
        self.time_of_impact = np.full((self.ny, self.nx), np.inf)
        # Use separate arrays to record blue and red influence
        self.blue_array = np.zeros((self.ny, self.nx))
        self.red_array = np.zeros((self.ny, self.nx))
        # Initialize blueness dictionary
        self.blueness_dict = {}
        # Initialize first decision
        self.segment_data = []
        self.last_wave_center = (self.player_x, self.player_y)  # Record last wave source position
        self._initialize_first_decision()
        
        # Add intercept coordinate attribute
        self.intercept_target = None  # Will be set by external function, format: (x, y)
    

    def _initialize_first_decision(self):
        """
        Initializes the first decision state and creates the initial wave segment.
        
        Args:
            None: Uses class attributes for initialization
        
        Returns:
            None: Updates internal state of the simulation
        """

        first_decision = self.decision_timeline[0]["decision"]
        self.current_decision = first_decision
        self._create_segment_info(0, self.last_wave_center, self.base_angle)
    

    def _update_decision(self, t):
        """
        Updates decision state based on current simulation time and decision timeline.
        Processes any pending decision changes or direction changes.
        
        Arg:
            t (float): Current simulation time
        
        Returns:
            None: Updates internal decision state and wave segments
        """

        # Check if decision switch is needed
        for decision in self.decision_timeline[:]:  # Create a copy for iteration
            time_point = decision["time"]
            if abs(t - time_point) < self.dt/2:  # Near time point
                if decision["decision"] == "change_face_direction":
                    self._handle_decision_change(t, decision)
                    self.decision_timeline.remove(decision)  # Remove processed decision
                elif decision["decision"] != self.current_decision:
                    self._handle_decision_change(t, decision)
                    self.decision_timeline.remove(decision)  # Remove processed decision
    

    def _handle_decision_change(self, t, decision):
        """
        Processes a decision change event, updating player direction and creating new wave segments.
        Manages transitions between different movement states.
        
        Args:
            t (float): Current simulation time
            decision (dict): Decision data containing type and optional angle parameter
        
        Returns:
            None: Updates player state and creates new wave segments
        """

        new_decision = decision["decision"]
        angle = decision.get("angle", None)
        # Get last wave segment info for calculating new starting point
        last_wave_center = self.last_wave_center

        if self.segment_data:
            last_segment = self.segment_data[-1]
            # Calculate midpoint of previous decision's wave arc as new wave source center
            last_wave_center = self._calculate_wave_center_at_time(last_segment, t)
            
            # Set end time of current segment to t
            self.segment_data[-1]["end_time"] = t
        
        if new_decision == "change_face_direction":
            if angle is not None:
                # Update base_angle to new angle
                self.base_angle = angle % 360
                
                # Update player position to new wave source center
                self.last_wave_center = last_wave_center
                self.player_x, self.player_y = last_wave_center
                
                # Clear current_decision
                self.current_decision = None

        else:
            # Handle normal decision
            old_decision = self.current_decision
            current_v = self._get_current_main_velocity()
            
            # Get new velocity dictionary
            v_dict = get_vmax_dict(new_decision)
            if v_dict:
                new_main_velocity = max(v_dict.values())
                if self.velocity_manager.start_transition(current_v, new_main_velocity, t):
                    print(f"At time {t}s: Transitioning from {old_decision} to {new_decision} with target velocity {new_main_velocity}.")
                else:
                    print(f"At time {t}s: Transition cancelled due to high velocity.")
            
            # Update decision and wave source center
            self.current_decision = new_decision
            self.last_wave_center = last_wave_center
            self.player_x, self.player_y = last_wave_center
            
            # Create new propagation segment
            self._create_segment_info(len(self.segment_data), last_wave_center, self.base_angle)
    

    def _get_current_main_direction_key(self):
        """
        Determines the current main propagation direction key based on active segment.
        
        Args:
            None: Uses current segment data
        
        Returns:
            str: Direction key (e.g., "forward", "left", etc.) representing main propagation direction
        """

        if not self.segment_data:
            return "forward"
        return self._dir_from_angle(
            self.segment_data[-1]["main_dir_global"],
            self.segment_data[-1]["base_angle"]
        )
    
    
    def _get_main_direction_for_decision(self, decision, base_angle):
        """
        Calculates the global angle for a given decision and base orientation.
        Maps player decisions to actual movement directions in global coordinates.
        
        Args:
            decision (str): Player movement decision
            base_angle (float): Current player orientation in global coordinates
        
        Returns:
            float: Global angle in degrees (0-360) representing main movement direction
        """

        if decision == "forward":
            return base_angle
        elif decision == "left-forward":
            return (base_angle - 45) % 360
        elif decision == "left":
            return (base_angle - 90) % 360
        elif decision == "left-behind":
            return (base_angle - 135) % 360
        elif decision == "behind":
            return (base_angle - 180) % 360
        elif decision == "right-behind":
            return (base_angle - 225) % 360
        elif decision == "right":
            return (base_angle - 270) % 360
        elif decision == "right-forward":
            return (base_angle - 315) % 360
        return base_angle
    

    def _create_segment_info(self, segment_index, wave_center, base_angle):
        """
        Creates a new wave propagation segment with specified parameters and timing information.
        Manages segment transitions and velocity distributions for wave propagation.
        
        Args:
            segment_index (int): Index of the new segment in the sequence
            wave_center (tuple): (x, y) coordinates of the wave origin point
            base_angle (float): Base orientation angle in global coordinates
        
        Returns:
            None: Updates segment_data list with new segment information
        """

        # Get current decision from decision_timeline
        if segment_index < len(self.decision_timeline):
            decision_entry = self.decision_timeline[segment_index]
            decision = decision_entry["decision"]
        else:
            decision = self.current_decision
        
        v_dict = get_vmax_dict(decision)
        
        # If v_dict is empty, it means the decision does not generate wave front (e.g., change_face_direction), skip
        if not v_dict:
            return
        
        # Calculate main propagation direction
        main_dir_global = self._get_main_direction_for_decision(decision, base_angle)
        
        # Calculate time
        if segment_index == 0:
            start_time = 0
        else:
            start_time = self.segment_data[-1]["end_time"]
        # Set end time of current segment to next decision time or t_end
        if segment_index + 1 < len(self.decision_timeline):
            end_time = self.decision_timeline[segment_index + 1]["time"]
        else:
            end_time = self.t_end
        
        info = {
            "segment_index": segment_index,
            "decision": decision,
            "wave_center": wave_center,
            "base_angle": base_angle,
            "main_dir_global": main_dir_global,
            "v_dict": v_dict,
            "start_time": start_time,
            "end_time": end_time
        }
        self.segment_data.append(info)
    

    def _initialize_grid(self):
        """
        Creates coordinate grid system for wave propagation simulation.
        Sets up X and Y meshgrids for calculating wave positions.
        
        Args:
            None: Uses class attributes for grid dimensions
        
        Returns:
            None: Initializes X and Y coordinate meshgrids as class attributes
        """

        self.xx = np.linspace(0, self.xmax, self.nx, endpoint=False)
        self.yy = np.linspace(0, self.ymax, self.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.xx, self.yy)
    

    def _get_global_angle(self, base_angle, local_dir):
        """
        Converts local direction references to global angle coordinates.
        Translates player-relative directions into absolute angles.
        
        Args:
            base_angle (float): Player's global orientation angle
            local_dir (str): Local direction key (e.g., "forward", "left-up")
        
        Returns:
            float: Global angle in degrees (0-360) for the specified direction
        """

        return (base_angle + local_direction_angles[local_dir]) % 360.0
    

    def _calculate_wave_center_at_time(self, segment, t):
        """
        Computes the wave source center position based on segment information and current time.
        Calculates new coordinates using velocity and direction parameters from the segment.
        
        Args:
            segment (dict): Wave segment containing start time, wave center, angles, and velocities
            t (float): Current simulation time to calculate the wave center position
        
        Returns:
            tuple: (x_new, y_new) coordinates of the wave center at time t, used for 
            tracking wave propagation and updating player position
        """

        start_time = segment["start_time"]
        # end_time is being set to t
        end_time = t
        wave_center = segment["wave_center"]
        base_angle = segment["base_angle"]
        main_dir_global = segment["main_dir_global"]
        v_dict = segment["v_dict"]
        
        # Calculate maximum velocity in main direction
        main_dir_key = self._dir_from_angle(main_dir_global, base_angle)
        v_max = v_dict.get(main_dir_key, 0)
        
        # Calculate propagation time
        propagated_time = end_time - start_time
        if propagated_time < 0:
            propagated_time = 0
        
        # Calculate propagation distance
        r_dir = self._calculate_propagation_distance(propagated_time, 0, v_max)
        
        # Calculate new center point coordinates
        rad = math.radians(main_dir_global)
        x_new = wave_center[0] + r_dir * math.sin(rad)
        y_new = wave_center[1] + r_dir * math.cos(rad)
        
        return (x_new, y_new)
    

    def _create_next_segment_info(self, segment_index):
        """
        Creates the next wave segment based on information from the previous segment.
        Calculates new wave source position and base angle from previous segment's endpoint.
        
        Arg:
            segment_index (int): Index of the next segment to be created, used to access previous segment data
        
        Returns:
            None: Updates segment_data list with new segment information and modifies wave propagation parameters
        """

        prev_info = self.segment_data[segment_index - 1]
        prev_center = prev_info["wave_center"]
        prev_main_dir_global = prev_info["main_dir_global"]
        # End time of previous segment:
        prev_end_time = prev_info["end_time"]
        
        # Calculate radius of previous segment (in main direction):
        main_dir_key = self._dir_from_angle(prev_main_dir_global, prev_info["base_angle"])
        v_max = prev_info["v_dict"][main_dir_key]
        r = self._calculate_propagation_distance(prev_end_time - prev_info["start_time"], 0, v_max)
        
        # Calculate new center point coordinates
        rad = math.radians(prev_main_dir_global)
        x_new = prev_center[0] + r * math.sin(rad)
        y_new = prev_center[1] + r * math.cos(rad)
        
        # Set the "base_angle" of next segment to the "main direction" global angle of previous segment
        next_base_angle = prev_main_dir_global
        # Generate segment info for next segment
        self._create_segment_info(segment_index, wave_center=(x_new, y_new), base_angle=next_base_angle)
    
    
    def _dir_from_angle(self, angle_global, base_angle):
        """
        Maps global angles to discrete direction categories in player's local coordinate system.
        Determines which of the eight cardinal/intercardinal directions best matches the angle.
        
        Args:
            angle_global (float): Angle in global coordinates
            base_angle (float): Player's base orientation angle
        
        Returns:
            str: Direction key ("forward", "left-up", etc.) matching the given angle
        """

        # If possible None or NaN, first do safety check
        if angle_global is None or base_angle is None:
            return "forward"  # Default fallback
        
        # Calculate relative angle
        angle_diff = (angle_global - base_angle) % 360
        # If angle_diff is NaN due to data anomaly, fall back
        if np.isnan(angle_diff):
            return "forward"
        
        # Iterate through 8 directions in local_direction_angles
        best_key = None
        min_diff = 999
        for k, v in local_direction_angles.items():
            d = abs((v - angle_diff) % 360)
            if d < min_diff:
                min_diff = d
                best_key = k
        
        # If no direction found or best_key is still None due to some anomaly, fall back to "forward"
        if best_key is None:
            return "forward"
        else:
            return best_key
    
    
    def _get_segment_for_time(self, t):
        """
        Retrieves the wave segment information active at a given time point.
        Returns the last segment if time exceeds simulation duration.
        
        Arg:
            t (float): Current simulation time
        
        Returns:
            dict: Wave segment information containing propagation parameters for the specified time
        """

        for seg in self.segment_data:
            if seg["start_time"] <= t < seg["end_time"]:
                return seg
        # If beyond the last segment
        return self.segment_data[-1] if self.segment_data else None
    

    def get_time_dependent_speed(self, t, base_v_direction):
        """
        Calculates instantaneous speed at time t using exponential acceleration model.
        Scales maximum velocity based on direction-specific base speed.
        
        Args:
            t (float): Current time
            base_v_direction (float): Base maximum velocity for the specified direction
        
        Returns:
            float: Instantaneous velocity (m/s) at time t for the given direction
        """

        ratio = base_v_direction / 11.0
        v_max = 11.0
        tau   = 1.2
        return ratio * v_max * (1 - np.exp(- t / tau))
    

    def _calculate_propagation_distance(self, t, start_t, v_max):
        """
        Computes wave propagation distance using exponential velocity model.
        Integrates velocity over time interval to determine total distance traveled.
        
        Args:
            t (float): Current time
            start_t (float): Start time of the propagation
            v_max (float): Maximum velocity for the direction
        
        Returns:
            float: Total distance traveled by wave front in the specified direction
        """
        
        T = t - start_t
        if T <= 0:
            return 0.0
        tau = 1.3  # time constant
        # d = v_max * [ T - tau + tau * exp(-T/tau) ]
        distance = v_max * ( T - tau + tau * math.exp(-T/tau) )
        return distance
    
    
    def _update_impact(self, t):
        """
        Updates wave impact on grid points for current time step.
        Calculates wave front positions and updates color intensities.
        
        Arg:
            t (float): Current simulation time
        
        Returns:
            None: Updates blue_array and red_array with new wave impact values
        """

        seg = self._get_segment_for_time(t)
        if not seg:
            return

        start_t = seg["start_time"]
        wave_center = seg["wave_center"]
        base_angle = seg["base_angle"]
        v_dict = seg["v_dict"]   

        dt_in_segment = t - start_t
        if dt_in_segment < 0:
            return

        dist = np.sqrt((self.X - wave_center[0])**2 + (self.Y - wave_center[1])**2)
        dx = self.X - wave_center[0]
        dy = self.Y - wave_center[1]
        raw_angle = np.arctan2(dx, dy)
        angle_deg = np.degrees(raw_angle) % 360

        wave_thickness = 0.5

        for dir_key, local_angle in local_direction_angles.items():
            global_angle = (base_angle + local_angle) % 360
            lower_deg = (global_angle - half_angle) % 360
            upper_deg = (global_angle + half_angle) % 360
            r_dir = self._calculate_propagation_distance(t - start_t, 0, v_dict.get(dir_key, 0))           
            if lower_deg > upper_deg:
                mask_angle = (angle_deg >= lower_deg) | (angle_deg <= upper_deg)
            else:
                mask_angle = (angle_deg >= lower_deg) & (angle_deg <= upper_deg)

            
            lower_r = r_dir - wave_thickness
            if lower_r < 0:
                lower_r = 0
            mask_dist = (dist >= lower_r) & (dist <= r_dir)

            mask = mask_angle & mask_dist
            self.blue_array[mask] = np.maximum(self.blue_array[mask], 1.0)

    
    def _update_colors(self, t):
        """
        Applies color decay and updates blueness tracking for visualization.
        Manages color intensity values and calculates total blueness metric.
        
        Arg:
            t (float): Current simulation time
        
        Returns:
            None: Updates color arrays and blueness_dict with new values
        """

        # 1. Apply decay to previously affected pixels
        self.blue_array = color_decay(self.blue_array)
        self.red_array = color_decay(self.red_array)
        
        # 2. Calculate blueness
        # blueness = blue value - red value
        blueness = self.blue_array - self.red_array
        # Set negative values in blueness to 0
        blueness = np.maximum(blueness, 0)
        # Calculate total blueness value
        total_blueness = np.sum(blueness)
        self.blueness_dict[round(t, 2)] = total_blueness
    

    def create_animation(self):
        """
        Creates and configures animation of wave propagation simulation.
        Sets up matplotlib figure and animation parameters.
        
        Args:
            None: Uses class attributes for animation configuration
        
        Returns:
            FuncAnimation: Matplotlib animation object ready for display or saving
        """

        fig, ax = plt.subplots(figsize=(12, 6))
        # Set up color mapping
        # Use overlapping effect of blue and red
        # Blue corresponds to positive values, red to negative values
        # Display using custom colormap
        cmap = plt.cm.RdBu  # Red-Blue colormap
        norm = plt.Normalize(-1.0, 1.0)
        
        # Initial color display as white
        combined_color = self.blue_array - self.red_array
        im = ax.imshow(combined_color,
                      origin='lower',
                      cmap=cmap,
                      norm=norm,
                      extent=[0, self.xmax, 0, self.ymax])
        
        ax.set_title("Wave Propagation Simulation with Blueness Tracking")
        ax.set_xlabel("X axis (→)")
        ax.set_ylabel("Y axis (↑)")
        
        # Add decision point markers
        for decision in self.decision_timeline:
            ax.axvline(x=decision["time"], color='r', linestyle='--', alpha=0.3)
        
        
        def update_frame(frame):
            """
            Updates the animation frame by calculating wave propagation and color values.
            Called by FuncAnimation for each frame of the simulation visualization.
            
            Arg:
                frame (int): Current frame number in the animation sequence
            
            Returns:
                list: List containing the updated image object for matplotlib animation,
                used by FuncAnimation to render the next frame
            """

            t = frame * self.dt
            if t > self.t_end:
                return [im]
            
            # Update decision
            self._update_decision(t)
            # Update wave front
            self._update_impact(t)
            # Update colors and blueness
            self._update_colors(t)
            
            # Update displayed color array
            combined_color = self.blue_array - self.red_array
            im.set_data(combined_color)
            return [im]
        
        anim = FuncAnimation(
            fig, update_frame,
            frames=self.n_frames + 1,
            interval=50,
            blit=True
        )
        
        return anim
    

    def _get_current_main_velocity(self):
        """
        Retrieves the current maximum velocity in the main movement direction.
        Used to determine speed limits for wave propagation calculations.
        
        Args:
            None: Uses current segment data from class attributes
        
        Returns:
            float: Current maximum velocity value in the main direction, used for 
            velocity transitions and wave propagation speed calculations
        """

        if not self.segment_data:
            return 0
            
        current_segment = self.segment_data[-1]
        # Fix here to get direction key instead of angle
        main_dir_key = self._dir_from_angle(
            current_segment["main_dir_global"],
            current_segment["base_angle"]
        )
        return current_segment["v_dict"][main_dir_key]
    

    def calculate_distance(self, point1, point2):
        """
        Calculates Euclidean distance between two points in 2D space.
        
        Args:
            point1 (tuple): (x, y) coordinates of first point
            point2 (tuple): (x, y) coordinates of second point
        
        Returns:
            float: Euclidean distance between the two points
        """

        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


##############################################################################
# 5) Running Example
##############################################################################
if __name__ == "__main__":
    # Note this is only for testing
    # you can change the nx ny pixel size to see different results
    #In general, the larger the pixel size, the more accurate the simulation
    # but also the more time it takes to run

    sim = WaveSimulation(
        xmax=120, ymax=53,
        nx=1200, ny=530,
        dt=0.1,
        t_end=10.0,
        initial_player_angle=270.0,
        start_x=15, start_y=20
    )
    
    animation = sim.create_animation()
    plt.show()
    # test output
    print("Blueness over time:")
    for t, blueness in sorted(sim.blueness_dict.items()):
        print(f"Time {t}s: Blueness = {blueness:.2f}")

