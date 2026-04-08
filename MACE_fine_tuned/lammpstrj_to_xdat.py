from ase.io import iread, write

input_file = "dump_first4000.lammpstrj"
output_file = "XDATCAR"

symbols = ["Li", "La", "Zr", "Co", "O"]

first = True
for atoms in iread(input_file, format="lammps-dump-text", specorder=symbols):
    if first:
        write(output_file, atoms, format="vasp-xdatcar")
        first = False
    else:
        write(output_file, atoms, format="vasp-xdatcar", append=True)
