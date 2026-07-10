# kg4vasp (VASP 6.4.0)

## Requirements

- VASP 6.4.0
- ASCII `nabla.dat` output (`-DNABLA1`)

## Patch VASP

Modify `src/optics.F` to enable `nabla.dat` output (VASP 6.4.0).

Then add the following preprocessor flag to `makefile.include`:

```make
CPP_OPTIONS += -DNABLA1
```

Recompile VASP:

```bash
make veryclean
make std
```

## Generate `nabla.dat`

Run the optical calculation with

```text
LOPTICS = .TRUE.
```

The current implementation has been verified using

```bash
mpirun -n 1 vasp_std
```

which generates an ASCII `nabla.dat`.

> **Note:** Multi-MPI execution currently does not generate `nabla.dat` correctly with the present implementation.

## Compile kg4vasp

```bash
cd src
make
```
