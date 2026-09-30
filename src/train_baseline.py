"""
Day 7-9 -- Baseline classifier: CSP + LDA (or SVM).

Loads the epoched, filtered trials produced by Day 4-6
(data/processed/*.fif), extracts Common Spatial Pattern (CSP) features, and
trains a linear classifier to distinguish rest / imagined-left / imagined-
right trials.

Why CSP first: it's the standard opening move in motor-imagery BCI
decoding. It learns spatial filters that maximize variance for one class
while minimizing it for the others -- exactly the spatial signature that
mu/beta ERD/ERS produces over motor cortex. It's fast, interpretable, and
a fair baseline to beat with the CNN in Day 10-13.

Run:
    python src/train_baseline.py --subjects 1
    python src/train_baseline.py --subjects 1 2 3 --classifier svm
"""

import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

N_CSP_COMPONENTS = 6


def load_epochs(subjects):
    all_epochs = []
    for subject in subjects:
        path = os.path.join(PROCESSED_DIR, f"subject{subject:03d}-epo.fif")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run "
                f"`python src/preprocess.py --subjects {subject}` first."
            )
        all_epochs.append(mne.read_epochs(path, preload=True, verbose=False))

    if len(all_epochs) == 1:
        return all_epochs[0]
    return mne.concatenate_epochs(all_epochs)


def build_pipeline(classifier: str) -> Pipeline:
    csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
    if classifier == "lda":
        clf = LinearDiscriminantAnalysis()
    elif classifier == "svm":
        clf = SVC(kernel="linear", probability=True)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return Pipeline([("CSP", csp), ("classifier", clf)])


def evaluate(pipeline: Pipeline, X, y, labels, tag: str) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    print(f"\n  5-fold cross-validated accuracy: {acc:.3f}")
    print("\n  Classification report:")
    print(classification_report(y, y_pred, target_names=labels, zero_division=0))

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {tag} (acc={acc:.2f})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")

    return acc


def main():
    parser = argparse.ArgumentParser(description="Day 7-9: CSP + LDA/SVM baseline classifier")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--classifier", choices=["lda", "svm"], default="lda")
    args = parser.parse_args()

    print(f"Loading epochs for subjects {args.subjects} ...")
    epochs = load_epochs(args.subjects)

    X = epochs.get_data(copy=False)  # (n_epochs, n_channels, n_times)
    label_map = epochs.event_id  # e.g. {'T0': 1, 'T1': 2, 'T2': 3}
    y = epochs.events[:, 2]
    labels = list(label_map.keys())

    print(f"Total trials: {len(y)}  |  classes: {label_map}")
    chance = max((y == v).sum() for v in label_map.values()) / len(y)
    print(f"Majority-class baseline (always guess the most common class): {chance:.3f}")

    pipeline = build_pipeline(args.classifier)
    tag = f"{args.classifier}_" + "-".join(f"{s:03d}" for s in args.subjects)

    acc = evaluate(pipeline, X, y, labels, tag)

    print("\nFitting final model on all data for saving ...")
    pipeline.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"baseline_{tag}.joblib")
    joblib.dump({"pipeline": pipeline, "label_map": label_map}, model_path)
    print(f"Saved model: {model_path}")

    print(f"\nDay 7-9 done: baseline {args.classifier.upper()} accuracy = {acc:.3f} "
          f"(majority-class baseline was {chance:.3f}).")
    print("Next (Day 10-13): deep learning classifier (CNN) on filtered epochs.")


if __name__ == "__main__":
    main()
