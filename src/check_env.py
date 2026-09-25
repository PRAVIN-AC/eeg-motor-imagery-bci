"""
Day 1 sanity check.
Run this after `pip install -r requirements.txt` to confirm every library
the pipeline depends on is importable, and that MNE can reach PhysioNet.

    python src/check_env.py
"""

import importlib
import sys

REQUIRED = ["mne", "numpy", "pandas", "sklearn", "matplotlib", "torch", "pybullet"]


def check_imports():
    print("Checking imports...")
    missing = []
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [ok]   {name:<12} {version}")
        except ImportError as e:
            missing.append(name)
            print(f"  [FAIL] {name:<12} {e}")
    return missing


def check_physionet_reachable():
    """
    Tries a lightweight MNE call that resolves the PhysioNet EEGMMI dataset
    index without downloading full recordings. If this fails, Day 2's
    dataset download will also fail, so we surface it now.
    """
    print("\nChecking PhysioNet EEG Motor Movement/Imagery access...")
    try:
        from mne.datasets import eegbci

        # This just resolves URLs / local cache paths, it does not pull
        # gigabytes of data. Safe to call at Day 1.
        path = eegbci.load_data(subject=1, runs=[1], update_path=True)
        print(f"  [ok] Sample run resolved/downloaded to: {path}")
    except Exception as e:
        print(f"  [FAIL] Could not reach/load PhysioNet EEGMMI data: {e}")
        return False
    return True


if __name__ == "__main__":
    missing = check_imports()
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    ok = check_physionet_reachable()
    if not ok:
        print("\nEnvironment libraries are fine, but dataset access failed.")
        print("Check your internet connection / firewall before Day 2.")
        sys.exit(1)

    print("\nDay 1 environment check PASSED. You're ready for Day 2 (data loading).")
