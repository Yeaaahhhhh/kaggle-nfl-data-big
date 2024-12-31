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

  A Simple program to calculate win rate for each team in the NFL accross all 9 weeks of the season.
  This program will read the games.csv file and calculate the win rate for each team.
  
Author: Xiangtian Dai   donktr17@gmail.com

Created: 10th Dec, 2024

'''


import csv
from collections import OrderedDict
from typing import Dict, List


def read_games_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Reads the games.csv file and returns a list of game records.

    Args:
        file_path (str): The path to the games.csv file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary represents a game.
    """
    
    games = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                games.append(row)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")

    return games


def initialize_team_stats() -> Dict[str, Dict[str, int]]:
    """
    Initializes the team statistics dictionary.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary with team abbreviations as keys and their stats as values.
    """

    return {}


def update_team_stats(team_stats: Dict[str, Dict[str, int]], game: Dict[str, str]) -> None:
    """
    Updates the team statistics based on a single game record.

    Args:
        team_stats (Dict[str, Dict[str, int]]): The current statistics of all teams.
        game (Dict[str, str]): A dictionary representing a single game.
    """

    home_team = game['homeTeamAbbr']
    visitor_team = game['visitorTeamAbbr']
    try:
        home_score = int(game['homeFinalScore'])
        visitor_score = int(game['visitorFinalScore'])
    except ValueError:
        print(f"Invalid score data in game: {game}")
        return

    for team in [home_team, visitor_team]:
        if team not in team_stats:
            team_stats[team] = {'wins': 0, 'total_games': 0, 'net_point_diff': 0}

    team_stats[home_team]['total_games'] += 1
    team_stats[visitor_team]['total_games'] += 1

    if home_score > visitor_score:
        team_stats[home_team]['wins'] += 1
    elif visitor_score > home_score:
        team_stats[visitor_team]['wins'] += 1
    # Draws do not count towards wins

    team_stats[home_team]['net_point_diff'] += (home_score - visitor_score)
    team_stats[visitor_team]['net_point_diff'] += (visitor_score - home_score)


def calculate_win_rates(team_stats: Dict[str, Dict[str, int]]) -> Dict[str, List[float]]:
    """
    Calculates the win rate and net point differential for each team.
    Rounds win rates to 2 decimal places.

    Args:
        team_stats (Dict[str, Dict[str, int]]): The statistics of all teams.

    Returns:
        Dict[str, List[float]]: A dictionary with team abbreviations as keys and a list containing
        win rate (2 decimal places) and net point differential as values.
    """

    final_stats = {}
    for team, stats in team_stats.items():
        total_games = stats['total_games']
        wins = stats['wins']
        net_diff = stats['net_point_diff']
        # Round win rate to 2 decimal places using round()
        win_rate = round(wins / total_games if total_games > 0 else 0, 2)
        final_stats[team] = [win_rate, net_diff]
    return final_stats


def sort_team_stats(final_stats: Dict[str, List[float]]) -> OrderedDict:
    """
    Sorts the team statistics first by win rate in descending order, then by net point differential in descending order.

    Args:
        final_stats (Dict[str, List[float]]): The final statistics of all teams.

    Returns:
        OrderedDict: An ordered dictionary sorted based on the specified criteria.
    """
    sorted_stats = sorted(
        final_stats.items(),
        key=lambda item: (-item[1][0], -item[1][1])
    )
    return OrderedDict(sorted_stats)


def get_winrate() -> Dict[str, List[float]]:
    """
    Processes the games.csv file and returns the win rate and net point differential for each team.

    Returns:
        Dict[str, List[float]]: An ordered dictionary with team abbreviations as keys and a list containing
        win rate and net point differential as values, sorted accordingly.
    """

    games = read_games_csv('games.csv')
    team_stats = initialize_team_stats()

    for game in games:
        update_team_stats(team_stats, game)

    final_stats = calculate_win_rates(team_stats)
    sorted_final_stats = sort_team_stats(final_stats)
    return sorted_final_stats


if __name__ == "__main__":
    winrate_dict = get_winrate()
    print(winrate_dict)
