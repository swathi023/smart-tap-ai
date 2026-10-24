"""
Smart Tap AI - Train & evaluate crowd-level classifier
-------------------------------------------------------
Run order:
    python src/dataset_generator.py --rows 3000 --out trips.csv
    python src/train_model.py --data trips.csv

Outputs:
    decision_tree_model.pkl   (used by hardware_bridge.py and the dashboard)
    confusion_matrix.png, feature_importance.png
"""

import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Must match crowd_optimization.predict_crowd_level exactly
FEATURES = ["hour", "day_of_week", "distance_km", "passenger_count_today"]
TARGET = "crowd_level"
LABELS = ["LOW", "MEDIUM", "HIGH"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="trips.csv")
    parser.add_argument("--out", default="decision_tree_model.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y = df[FEATURES], df[TARGET]
    print("Class balance:\n", y.value_counts(normalize=True).round(3), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Baselines, to justify calling the tree "optimized" ---
    baselines = {
        "Decision Tree (default)": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }
    print("=== Baselines (5-fold CV on train set) ===")
    for name, m in baselines.items():
        scores = cross_val_score(m, X_train, y_train, cv=5)
        print(f"{name:26s} {scores.mean():.4f} +/- {scores.std():.4f}")

    # --- Optimized Decision Tree ---
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 4, 5, 6, 8, 10, None],
            "min_samples_leaf": [1, 5, 10, 20],
        },
        cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("\nBest params:", grid.best_params_)

    pred = best.predict(X_test)
    print(f"\nTest accuracy (optimized tree): {accuracy_score(y_test, pred):.4f}\n")
    print(classification_report(y_test, pred, labels=LABELS))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=LABELS)
    plt.title("Confusion matrix - optimized Decision Tree")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    imp = pd.Series(best.feature_importances_, index=FEATURES).sort_values()
    imp.plot(kind="barh", title="Feature importance")
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump(best, args.out)
    print(f"[OK] Saved model to {args.out}")


if __name__ == "__main__":
    main()
