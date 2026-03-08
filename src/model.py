from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_trigger_model(df):
    """
    Model 1:
    Estimates P(early payout trigger)
    """

    X = df[[
        "odd_fav",
        "odd_dog",
        "lambda_fav",
        "lambda_dog"
    ]]

    y = df["early_payout_trigger"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)

    return model, roc, X_test


def train_comeback_model(df):
    """
    Model 2:
    Estimates P(favorite not win | early payout trigger = 1)
    Only trains on matches where early payout was triggered.
    """

    triggered_df = df[df["early_payout_trigger"] == 1].copy()

    X = triggered_df[[
        "odd_fav",
        "odd_dog",
        "lambda_fav",
        "lambda_dog"
    ]]

    y = triggered_df["not_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)

    return model, roc
