# Patch for vasp/src/optics.F
**patch -p1 < optics.patch** for ASCII version 
And to modify makefile.include add tag **CPP_OPTIONS += -DNABLA1**

# For kg4vasp compile
cd src
make
