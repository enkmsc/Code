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

material = "LATP_2Al"

# Base path
base_path = "/data01/tian_02/ML_LATP/MD/MLMD_1ns_1/LATP_2Al/LATP_2Al_md_600K/"
trajectory_path = base_path + f"{material}_md_out.traj"
log_path = base_path + f"logfile"
xdatcar_path = base_path + f"XDATCAR"
fine_tune_path = "/data01/tian_02/ML_LATP/MD/fine_tune_spin/LATP_2Al/03-22-2025/bestE_epoch19_e1_f68_s56_m3.pth.tar"
# Load the relaxed structure from the VASP file
structure_0 = Structure.from_file(base_path + "POSCAR")
structure_0.to(fmt="CONTCAR", filename=base_path+"chg.cif")
structure = Structure.from_file(base_path+"chg.cif")

# Load the CHGNet model
md_model = CHGNet.from_file(fine_tune_path)
# md_model = CHGNet.load()
# Create the MolecularDynamics instance using the relaxed structure
md = MolecularDynamics(
    atoms=structure,
    model=md_model,
    ensemble="nvt",
    starting_temperature=600,
    temperature=600,
    thermostat="Nose-Hoover",
    timestep=2,
    taut=10,
    trajectory=trajectory_path,
    loginterval=1,
    logfile=log_path,
    use_device="cuda"
)

# Run the molecular dynamics simulation
#md.run(25000)
md.run(501000)
#md.run(500)
#md.run(51000)

# After the simulation is complete, use ASE to read the .traj file
traj = Trajectory(trajectory_path)

# Save the final step's structure as a VASP CONTCAR file
final_structure = traj[-1]
write(base_path+ "CONTCAR", final_structure, format="vasp")
write(xdatcar_path, traj, format="vasp-xdatcar")
