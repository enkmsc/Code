from ase.io import read
from ase.io import write
from ase.io.trajectory import Trajectory
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from chgnet.model.dynamics import CHGNetCalculator
from pymatgen.io.xyz import XYZ
from ase.io import Trajectory, write
#from ase.io import read, write
import warnings

warnings.filterwarnings("ignore", module="pymatgen")

material = "LLTO"

# Base path
base_path = "BASEPATH"
trajectory_path = base_path + f"{material}_md_out.traj"
log_path = base_path + f"logfile"
xdatcar_path = base_path + f"XDATCAR"
# Load the relaxed structure from the VASP file
structure_0 = Structure.from_file(base_path + "POSCAR")
structure_0.to(fmt="CONTCAR", filename=base_path+"chg.cif")
structure = Structure.from_file(base_path+"chg.cif")

# Load the CHGNet model

md_model = CHGNet.load()
# Create the MolecularDynamics instance using the relaxed structure
md = MolecularDynamics(
    atoms=structure,
    model=md_model,
    ensemble="nvt",
    starting_temperature=TARGETTEMP,
    temperature=TARGETTEMP,
    thermostat="Nose-Hoover",
    timestep=2,
    taut=10,
    trajectory=trajectory_path,
    loginterval=1,
    logfile=log_path,
    use_device="cuda"
)

# Run the molecular dynamics simulation
md.run(1150000)

# After the simulation is complete, use ASE to read the .traj file
traj = Trajectory(trajectory_path)

# Save the final step's structure as a VASP CONTCAR file
final_structure = traj[-1]
write(base_path+ "CONTCAR", final_structure, format="vasp")
write(xdatcar_path, traj, format="vasp-xdatcar")
