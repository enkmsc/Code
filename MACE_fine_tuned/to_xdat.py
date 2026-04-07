from ase.io import iread, write

input_file = "md.traj"
output_file = "XDATCAR"

first = True

for atoms in iread(input_file):
    if first:
        write(output_file, atoms, format="vasp-xdatcar")
        first = False
    else:
        write(output_file, atoms, format="vasp-xdatcar", append=True)
