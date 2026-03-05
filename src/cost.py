import numpy as np


def expected_ep_cost(df, trigger_prob):

    stake = df["stake"].values
    odds = df["odd_fav"].values
    not_win = df["not_win"].values

    cost = np.sum(
        stake * odds * trigger_prob * not_win
    )

    return cost


def monte_carlo_cost(df, trigger_prob, n_sims=5000):

    stake = df["stake"].values
    odds = df["odd_fav"].values
    not_win = df["not_win"].values

    losses = []

    for _ in range(n_sims):

        trigger = np.random.binomial(1, trigger_prob)

        loss = np.sum(
            stake * odds * trigger * not_win
        )

        losses.append(loss)

    return np.array(losses)
