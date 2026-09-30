"""
Day 4-6 -- Preprocessing: filter to mu/beta band, epoch into labeled trials.

Turns the continuous raw EEG (Day 2-3 output) into a fixed-length,
band-limited Epochs object per subject: one epoch per T0 (rest) / T1
(imagined left fist) / T2 (imagined right fist) event, ready for feature
extraction and classification (Day 7-9).

Run:
    python src/preprocess.py --subjects 1
    python src/preprocess.py --subjects 1 2 3

Design choices (stated explicitly so the "why" survives into an interview):
  - Combined mu+beta band-pass (8-30 Hz): this is the standard band for
    motor-imagery decoding, because event-related (de)synchronization
    (ERD/ERS) spans both rhythms together over motor cortex. Splitting them
    into two separate bands is a reasonable later experiment, not required
    for a working Day 4-6 pipeline.
  - Epoch window: 0.5s to 3.5s after the cue onset. The first 0.5s is
    dropped to let the cue-onset visual/attention transient decay, so it
    doesn't dominate what the classifier learns.
  - No baseline correction on the saved epochs: CSP (Day 7-9) learns its
    own spatial filters from class covariance and does not need per-trial
    baselining. We DO baseline-correct a *separate* diagnostic plot below,
    purely to make the rest-vs-imagery comparison visually readable -- that
    plot is not used downstream.
"""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

warnings.filterwarnings("ignore", category=RuntimeWarning)

IMAGERY_RUNS = [4, 8, 12]
L_FREQ, H_FREQ = 8.0, 30.0  # combined mu + beta band
TMIN, TMAX = 0.5, 3.5  # epoch window relative to event onset (s), saved data

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_raw(subject: int, runs=IMAGERY_RUNS):
    fnames = eegbci.load_data(subject, runs, path=DATA_DIR, update_path=True)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in fnames]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn")
    return raw


def filter_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """Band-pass to the combined mu+beta band used for motor-imagery ERD/ERS."""
    return raw.copy().filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)


def make_epochs(raw_filt: mne.io.Raw):
    events, event_id = mne.events_from_annotations(raw_filt, verbose=False)
    picks = mne.pick_types(raw_filt.info, eeg=True)
    epochs = mne.Epochs(
        raw_filt,
        events,
        event_id=event_id,
        tmin=TMIN,
        tmax=TMAX,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs, event_id


def save_epochs(epochs: mne.Epochs, subject: int) -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, f"subject{subject:03d}-epo.fif")
    epochs.save(path, overwrite=True)
    print(f"  Saved epochs: {path}")
    return path


def plot_erd_sanity_check(raw_filt: mne.io.Raw, subject: int, event_id: dict):
    """
    Baselined, class-averaged waveform at a motor-cortex channel (C3/C4/Cz),
    as a visual sanity check that rest vs. imagined-movement trials actually
    look physiologically different. Not used downstream -- diagnostics only.
    """
    events, _ = mne.events_from_annotations(raw_filt, verbose=False)
    picks = mne.pick_types(raw_filt.info, eeg=True)
    epochs_baselined = mne.Epochs(
        raw_filt,
        events,
        event_id=event_id,
        tmin=-0.5,
        tmax=3.5,
        picks=picks,
        baseline=(-0.5, 0),
        preload=True,
        verbose=False,
    )

    ch_candidates = [c for c in ["C3", "C4", "Cz"] if c in epochs_baselined.ch_names]
    ch = ch_candidates[0] if ch_candidates else epochs_baselined.ch_names[0]
    ch_idx = epochs_baselined.ch_names.index(ch)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label in event_id:
        avg = epochs_baselined[label].average().data[ch_idx] * 1e6
        ax.plot(epochs_baselined.times, avg, label=label)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title(f"Subject {subject:03d} - {ch} - class-averaged filtered signal")
    ax.legend()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"subject{subject:03d}_erd_sanity_{ch}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Day 4-6: filter + epoch PhysioNet EEGMMI data")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    for subject in args.subjects:
        print(f"\n[subject {subject:03d}] loading raw ...")
        raw = load_raw(subject)

        print(f"  Filtering to {L_FREQ}-{H_FREQ} Hz (mu+beta) ...")
        raw_filt = filter_raw(raw)

        print("  Epoching ...")
        epochs, event_id = make_epochs(raw_filt)
        print(f"  Epochs: {len(epochs)} total, classes: {event_id}")
        for label in event_id:
            print(f"      {label:10s}: {len(epochs[label])}")

        save_epochs(epochs, subject)
        plot_erd_sanity_check(raw_filt, subject, event_id)

    print("\nDay 4-6 done: signal filtered to 8-30 Hz, epoched into labeled")
    print("rest/left/right trials, saved to data/processed/.")
    print("Next (Day 7-9): CSP feature extraction + LDA/SVM baseline classifier.")


if __name__ == "__main__":
    main()
