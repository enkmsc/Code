"""Edit these settings, then run: python evaluate_committee.py"""
# Relative paths below are relative to this settings.py file.
MODEL_PATHS = ["models/seed13.model", "models/seed42.model", "models/seed256.model"]
HEAD = None  # Set the same physical/DFT head for all models if needed.
DEVICE = "cuda"
DEFAULT_DTYPE = "float64"

INPUT_FILE = "input/dump.lammpstrj"
INPUT_FORMAT = "lammps-dump-text"  # Or "extxyz", "vasp", "lammps-data".
# LAMMPS type -> element. This example matches the uploaded LCO_LLZO_(2).data.
# Check your actual trajectory's pair_coeff element order before running!
TYPE_MAP = {1: "Li", 2: "La", 3: "Zr", 4: "Co", 5: "O"}
ATOM_STYLE = "atomic"  # Only used for lammps-data, e.g. "charge" if appropriate.

START_FRAME = 0  # Zero-based index in the input; not the LAMMPS timestep.
STOP_FRAME = None  # Exclusive upper bound.
FRAME_STRIDE = 100  # Every 100 saved frames, not every 100 MD steps.
MAX_FRAMES = 10  # Initial timing test. Set None to process all selected frames.

LOCAL_RADIUS = 4.0  # Angstrom; ranking neighborhood, NOT a crop/buffer radius.
RANK_BY = "local_mean"  # "local_mean" or "force_std".
CENTER_ELEMENTS = None  # None = all; e.g. ["Li", "Co", "O"].
MIN_SCORE = None  # eV/Angstrom; None = rank only, no calibrated threshold.
TOP_CENTERS_PER_FRAME = 10
MIN_CENTER_DISTANCE = 8.0  # Angstrom, minimum-image spacing within each frame.
OUTPUT_DIR = "results/committee_test"

