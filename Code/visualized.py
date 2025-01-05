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

Just a simple visualization to see distance change with time under formula V(t)


Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def compute_distance(t, V_max, tau):
    """
    Introduction:
    This function calculates the distance traveled over time by integrating 
    the velocity function V(t) = V_max * (1 - e^(-t/tau)). The distance 
    is given by the definite integral from 0 to t.

    Args:
        t (float or numpy.array): The time value(s) at which to compute the distance.
        V_max (float): The maximum velocity value.
        tau (float): The time constant for the exponential growth of velocity.

    Returns:
        float or numpy.array: The distance traveled at time t.
    """
    # Exact integral of V(t) from 0 to t:
    # ∫[0 to t] V_max * (1 - e^(-s/tau)) ds
    # = V_max * ( t - tau + tau * e^(-t / tau) )
    return V_max * (t - tau + tau * np.exp(-t / tau))



def init_animation():
    """
    Introduction:
    This function initializes the lines for the distance animation 
    by clearing any existing data.

    Args:
        None

    Returns:
        tuple: Contains the line objects (line1, line2) for distance.
    """
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2



def update_animation(frame):
    """
    Introduction:
    This function updates the data for both distance lines 
    (red for tau1 and blue for tau2) at each animation frame.

    Args:
        frame (int): The current frame index.

    Returns:
        tuple: The updated line1 and line2 objects.
    """
    # Current time
    t_current = frame * dt

    # Create time array from 0 to t_current for smooth line drawing
    times = np.linspace(0, t_current, frame + 1)

    # Compute distances for both tau values
    D1 = compute_distance(times, V_max, tau1)
    D2 = compute_distance(times, V_max, tau2)

    # Update lines
    line1.set_data(times, D1)
    line2.set_data(times, D2)

    return line1, line2


# -------------------- Main Code --------------------
if __name__ == "__main__":

    # Parameters
    V_max = 11
    tau1 = 1.1
    tau2 = 1.3

    # Animation settings
    dt = 0.01       # time step per frame
    T_final = 9.0   # total duration to simulate (seconds)
    num_frames = int(T_final / dt)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, T_final)
    
    # Since distance can grow up to around (V_max * (9 - tau)), let's set a safe upper bound
    ax.set_ylim(0, 110)

    # Set axis labels and title
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Distance (units)', fontsize=10)
    ax.set_title('Dynamic Distance Change with Time', fontsize=12)

    # Set axis ticks
    ax.set_xticks(np.arange(0, T_final + 1, 1))
    ax.set_yticks(np.arange(0, 111, 10))

    # Remove grid lines
    ax.grid(False)

    # Create empty line objects for the red and blue distance lines
    line1, = ax.plot([], [], color='red', linewidth=2, label='tau = 1.1')
    line2, = ax.plot([], [], color='blue', linewidth=2, label='tau = 1.3')

    # Add legend
    ax.legend(loc='upper left')

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update_animation,
        init_func=init_animation,
        frames=num_frames,
        interval=20,     # in milliseconds
        blit=False
    )

    # Save the animation as a GIF
    anim.save('distance.gif', writer='pillow')

    # plt.show()
