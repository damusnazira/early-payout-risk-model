import numpy as np


def expected_ep_cost(df, trigger_prob, comeback_prob):
    """
    Expected EP cost:
    stake × odds × P(trigger) × P(not win | trigger)
    """

    stake = df["stake"].values
    odds = df["odd_fav"].values

    cost = np.sum(
        stake * odds * trigger_prob * comeback_prob
    )

    return cost


def monte_carlo_cost(df, trigger_prob, comeback_prob, n_sims=5000):
    """
    Monte Carlo simulation of early payout cost:
    1. Simulate trigger
    2. Conditional on trigger, simulate comeback / not win
    """

    stake = df["stake"].values
    odds = df["odd_fav"].values

    losses = []

    for _ in range(n_sims):

        trigger = np.random.binomial(1, trigger_prob)
        comeback = np.random.binomial(1, comeback_prob)

        loss = np.sum(
            stake * odds * trigger * comeback
        )

        losses.append(loss)

    return np.array(losses)
