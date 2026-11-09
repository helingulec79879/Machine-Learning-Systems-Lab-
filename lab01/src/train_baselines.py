"""Step 2: train the two baselines and record test accuracy to 4 decimals.

Output: results/baseline_accuracy.csv
"""
import csv
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

from common import RESULTS_DIR, load_split, make_models, set_seeds


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"train={X_train.shape}, test={X_test.shape}")

    rows = []
    for name, factory in make_models().items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = factory().fit(X_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}", "converged": converged})
        print(f"{name:<20} accuracy={acc:.4f} converged={converged}")

    out = RESULTS_DIR / "baseline_accuracy.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_accuracy", "converged"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
