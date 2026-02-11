import os
import re
import numpy as np
from pymatgen.io.vasp import Vasprun

def magmom_outcar(outcar_path: str):
    """
    Return: list[float] length = N_atoms
    Reads the last 'magnetization (x)' block and returns the 'tot' column per atom.
    """
    if not os.path.exists(outcar_path):
        return None

    with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find all occurrences of "magnetization (x)"
    idxs = [i for i, line in enumerate(lines) if "magnetization (x)" in line]
    if not idxs:
        return None

    start = idxs[-1]  # last ionic step block
    # In VASP OUTCAR, after "magnetization (x)" there is a small header, then table.
    # We'll parse numeric rows until we hit a line starting with "tot".
    magmoms = []
    for j in range(start, len(lines)):
        s = lines[j].strip()

        if s.startswith("tot"):
            break

        # typical row: "  1    0.123    0.000    0.123"
        # numbers extraction (atom index + cols)
        if re.match(r"^\d+\s+[-\d\.Ee+]+\s+[-\d\.Ee+]+\s+[-\d\.Ee+]+", s):
            nums = re.findall(r"[-\d\.Ee+]+", s)
            # nums[0] is atom index, last column is 'tot'
            magmoms.append(float(nums[-1]))

    return magmoms if magmoms else None
