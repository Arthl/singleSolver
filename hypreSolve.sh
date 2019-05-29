# Go to hypreHOME
HYPREHOME=$1
cd $HYPREHOME

# Read solver parameters from file
read -d "\n" SOLVER NPROC PBSIZE SAVEMATSAS < solverPar.dat

# Call hypreSolver
mpirun -np $NPROC hypreMatlabSolve