'''
The MIT License (MIT)

CopyRight (c) 2024-2025 Xiangtian Dai

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
    and associated documentation files (the “Software”), to deal in the Software without restriction, 
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial 
    portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,  OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Define several functions based on essay Wide-Open-Space, calculate player's influence range and area for later calculation
and data visualization.

Author: Xiangtian Dai   donktr17@gmail.com
Created: 10th Dec, 2024

'''

import numpy as np


def calculate_influence(x, y, theta, s, ball_position):
    """ Calculate each players influence value

    Args:
        x: footballer's x coordinate (yard), numeric value
        y: footballer's y coordinate (yard), numeric value
        theta: footballer's orientation, numeric value in pi format
        s: speed (yard/s), numeric value
        ball_position: ball location in the field (x_b, y_b) (yard)

    return: 
        mu: mean value of the Gaussian distribution
        COV: covariance matrix of the Gaussian distribution
    """

    # Conversion angle: 0 degrees in the positive direction of the y-axis
    theta_math = (np.pi / 2) - theta  

    # Step 1: Calculate the radius of influence R_i(t)
    player_position = np.array([x, y])   
    ball_distance = np.linalg.norm(ball_position - player_position)
    R_i = 4 + 6 * min(1, ball_distance / 25)  # according to essay
    
    # Step 2: Calculate the speed scaling factor S_ratio
    S_ratio = s**2 / 11**2  # 11 yards/s max reachable
    
    # Step 3: Calculate the scaling matrix S and the rotation matrix R
    s_x = (R_i + R_i * S_ratio) / 2
    s_y = (R_i - R_i * S_ratio) / 2
    S = np.array([[s_x, 0], [0, s_y]])
    R_theta = np.array([
        [np.cos(theta_math), -np.sin(theta_math)],
        [np.sin(theta_math),  np.cos(theta_math)]
    ])
    
    # Covariance matrix COV
    COV = R_theta @ S @ S @ np.linalg.inv(R_theta)
    
    # Step 4: Calculate the mean μ_i(t)
    velocity_vector = np.array([s * np.cos(theta_math), s * np.sin(theta_math)])
    mu = player_position + 0.5 * velocity_vector
    
    return mu, COV


def multivariate_gaussian(pos, mu, cov):
    """ Calculate the value of a multivariate Gaussian distribution
        For data visualization purpose

    Args:
        pos: mesh grid position
        mu: mean value
        cov: covariace matrix

    return: 
        Z: will be visualize in mesh grid map
    """

    n = mu.shape[0]
    diff = pos - mu
    inv_cov = np.linalg.inv(cov)
    exponent = np.einsum('...k,kl,...l->...', diff, inv_cov, diff)
    norm_factor = 1 / (np.sqrt((2 * np.pi)**n * np.linalg.det(cov)))
    Z = norm_factor * np.exp(-0.5 * exponent)
    return Z


def gaussian_influence_area(x, y, theta, v, ball_position):
    """ Calculate influence area from a gaussian distribution
    
    Args:
        x: player's x-coordinate (yard)
        y: player's y-coordinate (yard)
        theta: the player's orientation
        v: player's speed (yard/s)
        ball_position: ball position (x_b, y_b) (yard)

    return: 
        area: numeric value, rounded 2 decimal
    """

    # calculate cov mainly
    mu, cov = calculate_influence(x, y, theta, v, ball_position)
    
    # calculate area
    det_cov = np.linalg.det(cov)
    area = np.pi * np.sqrt(det_cov)
    
    return round(area, 2)

