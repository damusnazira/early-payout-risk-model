import matplotlib.pyplot as plt
import numpy as np

from simulate_data import simulate_dataset
from model import train_model
from cost import expected_ep_cost, monte_carlo_cost
from breakeven import break_even_turnover, break_even_distribution


HOLD = 0.05


def main():

    # ---------------------------
    # 1. Generate simulated dataset
    # ---------------------------

    df = simulate_dataset(10000)

    # ---------------------------
    # 2. Train early payout model
    # ---------------------------

    model, roc, X_test = train_model(df)

    trigger_prob = model.predict_proba(X_test)[:, 1]

    print("Model ROC-AUC:", roc)

    # ---------------------------
    # 3. Estimate expected EP cost
    # ---------------------------

    df_test = df.iloc[X_test.index]

    ep_cost = expected_ep_cost(df_test, trigger_prob)

    print("Expected Early Payout Cost:", ep_cost)

    # ---------------------------
    # 4. Monte Carlo simulation
    # ---------------------------

    losses = monte_carlo_cost(df_test, trigger_prob)

    # ---------------------------
    # 5. Break-even turnover
    # ---------------------------

    break_even = break_even_turnover(ep_cost, HOLD)

    break_even_risk = break_even_distribution(losses, HOLD)

    print("Break-even incremental turnover:", break_even)

    print("Risk-adjusted break-even:", break_even_risk)

    # ---------------------------
    # 6. Plot loss distribution
    # ---------------------------

    plt.hist(losses, bins=40)

    plt.title("Early Payout Cost Distribution")

    plt.xlabel("Cost")

    plt.ylabel("Frequency")

    plt.show()


if __name__ == "__main__":

    main()
