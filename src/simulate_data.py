import numpy as np
import pandas as pd


def simulate_match_path(lambda_fav, lambda_dog, match_duration=90):
    """
    Simulates goal times for favorite and underdog using Poisson processes.
    Returns:
        early_payout_trigger: 1 if favorite leads by 2 goals at any point
        fav_win: 1 if favorite wins at full time
        goals_fav: final goals favorite
        goals_dog: final goals underdog
    """

    n_goals_fav = np.random.poisson(lambda_fav)
    n_goals_dog = np.random.poisson(lambda_dog)

    goal_times_fav = np.sort(np.random.uniform(0, match_duration, n_goals_fav))
    goal_times_dog = np.sort(np.random.uniform(0, match_duration, n_goals_dog))

    events = []

    for t in goal_times_fav:
        events.append((t, "fav"))

    for t in goal_times_dog:
        events.append((t, "dog"))

    events.sort(key=lambda x: x[0])

    fav_score = 0
    dog_score = 0
    early_payout_trigger = 0

    for _, team in events:
        if team == "fav":
            fav_score += 1
        else:
            dog_score += 1

        if fav_score - dog_score >= 2:
            early_payout_trigger = 1

    fav_win = int(fav_score > dog_score)

    return early_payout_trigger, fav_win, fav_score, dog_score


def simulate_dataset(n_matches=10000, seed=42):
    np.random.seed(seed)

    rows = []

    for _ in range(n_matches):

        # Pre-match strength / market variables
        lambda_fav = np.random.uniform(1.4, 2.4)
        lambda_dog = np.random.uniform(0.8, 1.8)

        odd_fav = np.random.uniform(1.3, 2.2)
        odd_dog = np.random.uniform(3.0, 6.0)

        stake = np.random.exponential(100)

        early_payout_trigger, fav_win, goals_fav, goals_dog = simulate_match_path(
            lambda_fav=lambda_fav,
            lambda_dog=lambda_dog,
            match_duration=90
        )

        not_win = 1 - fav_win

        rows.append({
            "stake": stake,
            "odd_fav": odd_fav,
            "odd_dog": odd_dog,
            "lambda_fav": lambda_fav,
            "lambda_dog": lambda_dog,
            "goals_fav": goals_fav,
            "goals_dog": goals_dog,
            "early_payout_trigger": early_payout_trigger,
            "not_win": not_win
        })

    df = pd.DataFrame(rows)
    return df
