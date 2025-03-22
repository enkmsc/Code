from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis
from pymatgen.core.trajectory import Trajectory

# 1200K 
xdat_path = "/Users/emilydai/LATP_2Al/"
traj = Trajectory.from_file(xdat_path+"XDATCAR_900K")
diff = DiffusionAnalyzer.from_structures(traj, specie='Li', temperature=900, time_step=2, step_skip=200)
pda = ProbabilityDensityAnalysis.from_diffusion_analyzer(diff, interval=0.5, species=("Li"))
pda.to_chgcar(xdat_path+"Traj_900K.vasp")
