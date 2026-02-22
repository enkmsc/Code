from __future__ import annotations

import re
from typing import TYPE_CHECKING

from monty.io import reverse_readfile
from pymatgen.io.vasp.outputs import Oszicar, Vasprun

if TYPE_CHECKING:
    from pymatgen.core import Structure


def parse_vasp_dir(
    file_root: str, check_electronic_convergence: bool = False
) -> dict[str, list]:
    """Parse VASP output files into structures and labels
    By default, the magnetization is read from mag_x from VASP,
    plz modify the code if magnetization is for (y) and (z).

    Args:
        file_root (str): the directory of the VASP calculation outputs
        check_electronic_convergence (bool): if set to True, this function will raise
            Exception to VASP calculation that did not achieve
    """
    try:
        oszicar = Oszicar(f"{file_root}/OSZICAR")
        vasprun_orig = Vasprun(
            f"{file_root}/vasprun.xml",
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
            exception_on_bad_xml=False,
        )
        outcar_filename = f"{file_root}/OUTCAR"
    except Exception:
        oszicar = Oszicar(f"{file_root}/OSZICAR.gz")
        vasprun_orig = Vasprun(
            f"{file_root}/vasprun.xml.gz",
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
            exception_on_bad_xml=False,
        )
        outcar_filename = f"{file_root}/OUTCAR.gz"


    n_atoms = len(vasprun_orig.ionic_steps[0]["structure"])

    dataset = {
        "structure": [],
        "uncorrected_total_energy": [],
        "energy_per_atom": [],
        "force": [],
        "stress": None if "stress" not in vasprun_orig.ionic_steps[0] else [],
    }

    for ionic_step in vasprun_orig.ionic_steps:
        #if (len(ionic_step["electronic_steps"]) >= vasprun_orig.parameters["NELM"]
        #):
        #    continue
        if check_electronic_convergence and (len(ionic_step["electronic_steps"]) >= vasprun_orig.parameters["NELM"]):
            continue

        dataset["structure"].append(ionic_step["structure"])
        dataset["uncorrected_total_energy"].append(ionic_step["e_wo_entrp"])
        dataset["energy_per_atom"].append(ionic_step["e_0_energy"] / n_atoms)
        dataset["force"].append(ionic_step["forces"])
        if "stress" in ionic_step:
            dataset["stress"].append(ionic_step["stress"])
        
    if dataset["uncorrected_total_energy"] == []:
        raise Exception(f"No data parsed from {file_root}!")

    return dataset
