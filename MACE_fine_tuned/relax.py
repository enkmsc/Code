from ase.io import read, write
from ase.optimize import FIRE
from ase import filters
from mace.calculators import MACECalculator

base_path = "BASEPATH"
atoms = read("STRUCTURE")

calc = MACECalculator(
        model_paths="MODEL",
        device="cuda",
        default_dtype="float64")

atoms.calc = calc

atoms_filter = filters.FrechetCellFilter(atoms)

opt = FIRE(
    atoms_filter,
    trajectory="relax.traj",
    logfile="logfile"
)

opt.run(fmax=0.03, steps=10000000)

write("CONTCAR", atoms, format="vasp")
