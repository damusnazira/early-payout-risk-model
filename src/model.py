from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_model(df):

    X = df[[
        "odd_fav",
        "odd_dog",
        "goals_fav",
        "goals_dog"
    ]]

    y = df["early_payout_trigger"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, probs)

    return model, roc, X_test
