"""
Day 2-3 — Load and explore the PhysioNet EEG Motor Movement/Imagery dataset.

What this does:
  1. Downloads a handful of subjects' EDF runs (cached locally after first run).
  2. Loads them with MNE, applies the standard 10-05 montage.
  3. Prints a summary: channels, sampling rate, duration, event counts.
  4. Saves a few diagnostic plots to results/ so you can *see* real EEG.

Run:
    python src/load_data.py
    python src/load_data.py --subjects 1 2 3 --show-plots

Notes on the PhysioNet EEGMMI run numbering (per-subject, 14 runs each):
    Runs 1, 2        -> baseline (eyes open, eyes closed)
    Runs 3, 7, 11     -> executed real left/right fist movement
    Runs 4, 8, 12     -> IMAGINED left/right fist movement   <- what we want
    Runs 5, 9, 13     -> executed real fists/feet movement
    Runs 6, 10, 14     -> imagined fists/feet movement
For left-hand-vs-right-hand motor IMAGERY (the classic BCI task this project
targets), we use runs 4, 8, 12.
"""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # write plots to file; no GUI needed / no display required
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Imagined left-hand vs right-hand fist movement.
IMAGERY_RUNS = [4, 8, 12]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_subject(subject: int, runs=IMAGERY_RUNS):
    """Download (if needed) and load one subject's motor-imagery runs as a
    single concatenated Raw object, with standard channel names/montage."""
    print(f"\n[subject {subject:03d}] fetching runs {runs} ...")
    fnames = eegbci.load_data(subject, runs, path=DATA_DIR, update_path=True)

    raws = [read_raw_edf(f, preload=True, verbose=False) for f in fnames]
    raw = concatenate_raws(raws)

    # PhysioNet channel names have trailing dots (e.g. "Fc5."); MNE's
    # standard montage expects names without them.
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn")

    return raw


def summarize(raw: mne.io.Raw, subject: int):
    print(f"\n=== Subject {subject:03d} summary ===")
    print(f"  Channels        : {len(raw.ch_names)}  ({raw.ch_names[:5]}...)")
    print(f"  Sampling rate   : {raw.info['sfreq']} Hz")
    print(f"  Duration        : {raw.times[-1]:.1f} s")

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print(f"  Event labels    : {event_id}")
    print(f"  Event counts    : {len(events)} total events")
    for label, code in event_id.items():
        count = (events[:, 2] == code).sum()
        print(f"      {label:10s}: {count}")

    return events, event_id


def save_plots(raw: mne.io.Raw, subject: int, show_plots: bool):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Raw time-series snippet (first 5 channels, first 10 seconds)
    fig1 = raw.copy().pick(raw.ch_names[:5]).plot(
        duration=10, n_channels=5, show=False
    )
    path1 = os.path.join(RESULTS_DIR, f"subject{subject:03d}_raw_snippet.png")
    fig1.savefig(path1, dpi=120)
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # 2) Power spectral density — should show visible mu (8-12Hz) / beta
    #    (13-30Hz) activity, which is exactly what Day 4-6 filtering targets.
    fig2 = raw.compute_psd(fmax=40).plot(show=False)
    path2 = os.path.join(RESULTS_DIR, f"subject{subject:03d}_psd.png")
    fig2.savefig(path2, dpi=120)
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # 3) Sensor layout, so you can see the scalp montage mapped correctly
    fig3 = raw.plot_sensors(show_names=False, show=False)
    path3 = os.path.join(RESULTS_DIR, f"subject{subject:03d}_sensors.png")
    fig3.savefig(path3, dpi=120)
    plt.close(fig3)
    print(f"  Saved: {path3}")

    if show_plots:
        print("  (--show-plots was set, but this environment saves to file "
              "instead of opening windows — open the PNGs in results/.)")


def main():
    parser = argparse.ArgumentParser(description="Day 2-3: load & explore PhysioNet EEGMMI data")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1],
                         help="Subject numbers to load, e.g. --subjects 1 2 3")
    parser.add_argument("--show-plots", action="store_true",
                         help="(plots are always saved to results/; this flag is a no-op reminder)")
    args = parser.parse_args()

    for subject in args.subjects:
        raw = load_subject(subject)
        summarize(raw, subject)
        save_plots(raw, subject, args.show_plots)

    print("\nDay 2-3 done: dataset loads, montage applies cleanly, "
          "events are labeled, and diagnostic plots are in results/.")
    print("Next (Day 4-6): band-pass filter to mu/beta and segment into "
          "labeled trial windows.")


if __name__ == "__main__":
    main()
