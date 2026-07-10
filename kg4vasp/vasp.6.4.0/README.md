## Requirements

- VASP 6.4.0
- ASCII `nabla.dat` output (`-DNABLA1`)

## Patch VASP 6.4.0 for ASCII version

Modify `src/optics.F` to enable `nabla.dat` output

`patch -p1 < optics.patch`

Then add the following preprocessor flag to `makefile.include`:

```make
CPP_OPTIONS += -DNABLA1
```
> **Note:** Multi-MPI execution currently does not generate `nabla.dat` correctly with the present implementation.

`mpirun -n 1 /where/is/bin/vasp_std > ***-out`

## Compile kg4vasp

```bash
cd src
make
```
