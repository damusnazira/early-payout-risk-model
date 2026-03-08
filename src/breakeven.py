import numpy as np


def break_even_turnover(ep_cost, hold):
    return ep_cost / hold


def break_even_distribution(losses, hold):

    expected = np.mean(losses)
    var95 = np.quantile(losses, 0.95)

    return {
        "expected_break_even": expected / hold,
        "var95_break_even": var95 / hold
    }
