# EEG Motor-Imagery Brain-Computer Interface

Decoding imagined movement from real human EEG and using the decoded intent
to drive a downstream action (cursor / simulated robotic arm) — a
non-invasive analogue of the signal → decode → action pipeline used by
BCI companies such as Neuralink.

**Dataset:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (public, free)
**Stack:** MNE-Python, PyTorch, scikit-learn, PyBullet

## Pipeline

1. **Data acquisition** — load the PhysioNet EEGMMI dataset (multi-channel EEG).
2. **Preprocessing** — band-pass filter to mu (8–12 Hz) and beta (13–30 Hz) bands.
3. **Feature extraction** — Common Spatial Patterns (CSP), or raw windows for end-to-end learning.
4. **Classification** — CSP+LDA/SVM baseline, then a CNN; optional spiking-network variant (snnTorch).
5. **Action mapping** — decoded class drives a cursor or a simulated PyBullet arm joint.
6. **Evaluation** — accuracy, confusion matrix, cross-subject generalization, and end-to-end decode latency.

## 20-Day Plan

| Days | Milestone |
|---|---|
| 1 | Environment setup (this step) |
| 2–3 | Load and explore the dataset |
| 4–6 | Preprocessing pipeline |
| 7–9 | Baseline classifier (CSP + LDA/SVM) |
| 10–13 | Deep learning classifier (CNN) |
| 14–16 | Action mapping demo |
| 17–18 | Latency & cross-subject robustness testing |
| 19–20 | Documentation & publishing |

## Day 1 — Environment Setup

```bash
bash setup_day1.sh
```

This creates a virtual environment, installs everything in `requirements.txt`,
runs `src/check_env.py` to confirm every library imports correctly and that
PhysioNet is reachable, then makes the first git commit.

If you'd rather do it by hand:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/check_env.py
git init && git add . && git commit -m "Day 1: environment setup"
```

Then create an empty repo on GitHub and push:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## Honest scope note

This is an **offline, non-invasive** decoding pipeline on pre-recorded data —
not a live neural interface. EEG has a much lower signal-to-noise ratio than
invasive recording, so expect real accuracy ceilings below what implanted
systems achieve. That gap is stated here on purpose: presenting it honestly
is part of what makes this a credible BCI-adjacent portfolio piece rather
than an overreach.
