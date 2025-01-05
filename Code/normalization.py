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

Just use cut cut cut ways to group up dataset into 4, and normalize the opportunity data

Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import orjson
import numpy as np
from typing import Any, Dict, List, Tuple


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The parsed JSON data as a dictionary.
    """
    with open(file_path, 'rb') as f:
        data = orjson.loads(f.read())
    return data


def collect_derivatives(data: Dict[str, Any]) -> List[float]:
    """
    Collect all derivative values from the nested JSON data.

    Args:
        data (Dict[str, Any]): The nested JSON data.

    Returns:
        List[float]: A list of all derivative values.
    """
    derivatives = []
    for game_play, players in data.items():
        for nfl_id, derivative_list in players.items():
            derivatives.extend(derivative_list)
    return derivatives


def compute_normalization_params(derivatives: List[float]) -> Tuple[float, float]:
    """
    Compute the min and max values for normalization based on the IQR method.

    Args:
        derivatives (List[float]): A list of derivative values.

    Returns:
        Tuple[float, float]: A tuple containing min_val and max_val for normalization.
    """
    q1 = np.percentile(derivatives, 25)
    q3 = np.percentile(derivatives, 75)
    iqr = q3 - q1
    min_val = q1 - 1.5 * iqr
    max_val = q3 + 1.5 * iqr
    return min_val, max_val


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a single derivative value to the range [0, 1].

    Args:
        value (float): The derivative value to normalize.
        min_val (float): The minimum value for normalization.
        max_val (float): The maximum value for normalization.

    Returns:
        float: The normalized value within [0, 1], rounded to 4 decimal places.
    """
    if value < min_val:
        return 0.0
    elif value > max_val:
        return 1.0
    else:
        return round(float((value - min_val) / (max_val - min_val)), 4)


def normalize_data(data: Dict[str, Any], min_val: float, max_val: float) -> Dict[str, Any]:
    """
    Normalize all derivative values in the JSON data.

    Args:
        data (Dict[str, Any]): The nested JSON data.
        min_val (float): The minimum value for normalization.
        max_val (float): The maximum value for normalization.

    Returns:
        Dict[str, Any]: The JSON data with normalized derivative values.
    """
    normalized_data = {}

    for game_play, players in data.items():
        normalized_players = {}
        for nfl_id, derivative_list in players.items():
            normalized_derivatives = [
                normalize_value(value, min_val, max_val) for value in derivative_list
            ]
            normalized_players[nfl_id] = normalized_derivatives
        normalized_data[game_play] = normalized_players

    return normalized_data


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save the JSON data to a file using orjson for faster serialization.

    Args:
        data (Dict[str, Any]): The JSON data to save.
        file_path (str): The path to the output JSON file.
    """
    # Convert numpy.float64 to Python float
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.float64):
            return float(obj)
        return obj

    converted_data = convert_to_native_types(data)
    with open(file_path, 'wb') as f:
        f.write(orjson.dumps(converted_data, option=orjson.OPT_INDENT_2))


def main() -> None:
    """
    Main function to load, normalize, and save the JSON data.
    """
    input_file = 'opportunity.json'
    output_file = 'normalized.json'

    # Load the original JSON data
    print("Loading JSON data...")
    data = load_json(input_file)

    # Collect all derivative values
    print("Collecting derivative values...")
    derivatives = collect_derivatives(data)

    # Compute normalization parameters
    print("Computing normalization parameters...")
    min_val, max_val = compute_normalization_params(derivatives)
    print(f"Normalization range: min_val = {min_val}, max_val = {max_val}")

    # Normalize the data
    print("Normalizing data...")
    normalized_data = normalize_data(data, min_val, max_val)

    # Save the normalized JSON data
    print("Saving normalized JSON data...")
    save_json(normalized_data, output_file)

    print("Normalization complete. Saved to 'normalized.json'.")


if __name__ == "__main__":
    main()
