import numpy as np
import pandas as pd


def simulate_dataset(n_matches=10000, seed=42):

    np.random.seed(seed)


    odd_fav = np.random.uniform(1.3, 2.2, n_matches)
    odd_dog = np.random.uniform(3, 6, n_matches)

    stake = np.random.exponential(100, n_matches)

    # Match dynamics (Poisson goals)


    goals_fav = np.random.poisson(1.8, n_matches)
    goals_dog = np.random.poisson(1.2, n_matches)

    # Early payout rule
    # Example: favorite leads by 2 goals


    early_payout_trigger = (goals_fav - goals_dog >= 2).astype(int)

    # Final outcome

    fav_win = (goals_fav > goals_dog).astype(int)

    # not_win is used in cost estimation
    not_win = 1 - fav_win

    df = pd.DataFrame({
        "stake": stake,
        "odd_fav": odd_fav,
        "odd_dog": odd_dog,
        "goals_fav": goals_fav,
        "goals_dog": goals_dog,
        "early_payout_trigger": early_payout_trigger,
        "not_win": not_win
    })

    return df
