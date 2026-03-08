import matplotlib.pyplot as plt
import numpy as np

from simulate_data import simulate_dataset
from model import train_trigger_model, train_comeback_model
from earlypayout_cost import expected_ep_cost, monte_carlo_cost
from breakeventurnover import break_even_turnover, break_even_distribution


HOLD = 0.05


def main():

    # 1. Generate simulated dataset
    df = simulate_dataset(10000)

    print("Early payout trigger rate:", df["early_payout_trigger"].mean())
    print("Not win rate:", df["not_win"].mean())

    # 2. Train trigger model
    trigger_model, trigger_roc, X_test = train_trigger_model(df)
    trigger_prob = trigger_model.predict_proba(X_test)[:, 1]

    print("Trigger model ROC-AUC:", trigger_roc)
    print("Mean predicted trigger probability:", trigger_prob.mean())

    # 3. Train comeback model
    comeback_model, comeback_roc = train_comeback_model(df)

    print("Comeback model ROC-AUC:", comeback_roc)

    # 4. Prepare test dataset
    df_test = df.iloc[X_test.index].copy()

    X_test_comeback = df_test[[
        "odd_fav",
        "odd_dog",
        "lambda_fav",
        "lambda_dog"
    ]]

    comeback_prob = comeback_model.predict_proba(X_test_comeback)[:, 1]

    print("Mean predicted comeback probability:", comeback_prob.mean())

    # 5. Estimate expected EP cost
    ep_cost = expected_ep_cost(df_test, trigger_prob, comeback_prob)

    print("Expected Early Payout Cost:", ep_cost)

    # 6. Monte Carlo simulation
    losses = monte_carlo_cost(df_test, trigger_prob, comeback_prob)

    # 7. Break-even turnover
    break_even = break_even_turnover(ep_cost, HOLD)
    break_even_risk = break_even_distribution(losses, HOLD)

    print("Break-even incremental turnover:", break_even)
    print("Risk-adjusted break-even:", break_even_risk)

    # 8. Plot cost distribution
    plt.figure(figsize=(8, 5))
    plt.hist(losses, bins=40)
    plt.axvline(np.mean(losses), color="red", linestyle="--", label="Expected Cost")
    plt.axvline(np.quantile(losses, 0.95), color="orange", linestyle="--", label="95% VaR")
    plt.title("Monte Carlo Distribution of Early Payout Cost")
    plt.xlabel("Early Payout Cost")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/ep_cost_distribution.png", dpi=300)
    plt.show()

    # 9. Plot break-even vs hold
    holds = [0.03, 0.05, 0.08, 0.10]
    turnovers = [break_even_turnover(ep_cost, h) for h in holds]

    plt.figure(figsize=(8, 5))
    plt.plot(holds, turnovers, marker="o")
    plt.title("Break-even Incremental Turnover vs Hold")
    plt.xlabel("Hold")
    plt.ylabel("Required Incremental Turnover")
    plt.tight_layout()
    plt.savefig("reports/figures/breakeven_vs_hold.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()


if __name__ == "__main__":

    main()
