/*
 Read a linear system assembled in Matlab
 Solve the linear system using hypre
 Interface: Linear-Algebraic (IJ)
 Available solvers: 0  - AMG (default)
                    1  - AMG-PCG
                    8  - ParaSails-PCG
                    50 - PCG
                    61 - AMG-FlexGMRES

 Copyright: M. Giacomini (2017)
 (based on ex5.c in the hypre examples folder)
*/

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include <time.h> 

#include "vis.c"

int main (int argc, char *argv[])
{
   clock_t begin = clock();
   int time_index;

   HYPRE_Int i;
   HYPRE_Int iErr = 0;

    
   HYPRE_Int myid, num_procs, dummy;
   HYPRE_Int solver_id;
    
   HYPRE_Int first_local_row, last_local_row, local_num_rows;
   HYPRE_Int first_local_col, last_local_col, local_num_cols;
    
   HYPRE_Real *values;

   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJVector ij_b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector ij_x;
   HYPRE_ParVector par_x;

   char saveMatsAs;
   
   FILE *fset;
    
   void *object;
    
   HYPRE_Solver solver, precond;
    

    /* Read dimension of the system and chosen solver from file */
    fset = fopen("solverPar.dat","r"); // read mode
    if( fset == NULL )
    {
        perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }
    hypre_fscanf(fset, "%d\n", &solver_id);
    hypre_fscanf(fset, "%d\n", &num_procs);
    hypre_fscanf(fset, "%d\n", &dummy);
    hypre_fscanf(fset, "%c\n", &saveMatsAs);
    fclose(fset);

    
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    
   /* Read the matrix previously assembled
      <filename>  = IJ.A.out to read in what has been printed out (processor numbers are omitted). */
      iErr = HYPRE_IJMatrixRead( "matrixK", MPI_COMM_WORLD, HYPRE_PARCSR, &ij_A );


    if (iErr) 
    {
        hypre_printf("ERROR: Problem reading in the system matrix!\n");
        exit(1);
    }
    /* Get dimension info */
    iErr = HYPRE_IJMatrixGetLocalRange( ij_A,
                                       &first_local_row, &last_local_row ,
                                       &first_local_col, &last_local_col );
    
    local_num_rows = last_local_row - first_local_row + 1;
    local_num_cols = last_local_col - first_local_col + 1;
   /* Get the parcsr matrix object to use */
   iErr += HYPRE_IJMatrixGetObject( ij_A, &object);
   parcsr_A = (HYPRE_ParCSRMatrix) object;
    
    
   /*  Read the RHS previously assembled */
   iErr = HYPRE_ParVectorRead(MPI_COMM_WORLD, "vectorF.0", &par_b);
   if (iErr)
   {
       hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
       exit(1);
   }
   ij_b = NULL;
    
    
   /* Create the initial solution and set it to zero */
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
   HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_x);
   /* Initialize the guess vector */
   values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
   for (i = 0; i < local_num_cols; i++)
      values[i] = 0.;
   HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
   hypre_TFree(values);
   /* Get the parcsr vector object to use */
   iErr = HYPRE_IJVectorGetObject( ij_x, &object );
   par_x = (HYPRE_ParVector) object;


   /* Choose a solver and solve the system */

   /* AMG */
   if (solver_id == 0)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_BoomerAMGCreate(&solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_BoomerAMGSetPrintLevel(solver, 3);  /* print solve info + parameters */
      HYPRE_BoomerAMGSetOldDefault(solver); /* Falgout coarsening with modified classical interpolaiton */
      HYPRE_BoomerAMGSetRelaxType(solver, 3);   /* G-S/Jacobi hybrid relaxation */
      HYPRE_BoomerAMGSetRelaxOrder(solver, 1);   /* uses C/F relaxation */
      HYPRE_BoomerAMGSetNumSweeps(solver, 1);   /* Sweeeps on each level */
      HYPRE_BoomerAMGSetMaxLevels(solver, 20);  /* maximum number of levels */
      HYPRE_BoomerAMGSetTol(solver, 1e-7);      /* conv. tolerance */

      time_index = hypre_InitializeTiming("AMG Setup and Solve");
      hypre_BeginTiming(time_index);
      /* Now setup and solve! */
      HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);

       hypre_EndTiming(time_index);
       hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      HYPRE_BoomerAMGDestroy(solver);
   }
   /* PCG */
   else if (solver_id == 50)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* prints out the iteration info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      time_index = hypre_InitializeTiming("PCG Setup and Solve");
      hypre_BeginTiming(time_index);
      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

       hypre_EndTiming(time_index);
       hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      HYPRE_ParCSRPCGDestroy(solver);
   }
   /* PCG with AMG preconditioner */
   else if (solver_id == 1)
   {
      int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);
      /* Now set up the AMG preconditioner and specify any parameters */
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetOldDefault(precond); 
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

      /* Set the PCG preconditioner */
      HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

	hypre_EndTiming(time_index);
	hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
	hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();

        time_index = hypre_InitializeTiming("PCG Solve");
        hypre_BeginTiming(time_index);

      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

       hypre_EndTiming(time_index);
       hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver and preconditioner */
      HYPRE_ParCSRPCGDestroy(solver);
      HYPRE_BoomerAMGDestroy(precond);
   }
   /* PCG with Parasails Preconditioner */
   else if (solver_id == 8)
   {
      int    num_iterations;
      double final_res_norm;

      int      sai_max_levels = 1;
      double   sai_threshold = 0.1;
      double   sai_filter = 0.05;
      int      sai_sym = 1;

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      time_index = hypre_InitializeTiming("ParaSails Setup");
      hypre_BeginTiming(time_index);
      /* Now set up the ParaSails preconditioner and specify any parameters */
      HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
      HYPRE_ParaSailsSetFilter(precond, sai_filter);
      HYPRE_ParaSailsSetSym(precond, sai_sym);
      HYPRE_ParaSailsSetLogging(precond, 3);

      /* Set the PCG preconditioner */
      HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup, precond);

	hypre_EndTiming(time_index);
	hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
	hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();

        time_index = hypre_InitializeTiming("PCG Solve");
        hypre_BeginTiming(time_index);

      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

       hypre_EndTiming(time_index);
       hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();


      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      HYPRE_ParCSRPCGDestroy(solver);
      HYPRE_ParaSailsDestroy(precond);
   }
/* PCG with Euclid Preconditioner */
   else if (solver_id == 10)
   {
      int    num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_PCGSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

      time_index = hypre_InitializeTiming("Euclid Setup");
      hypre_BeginTiming(time_index);
      /* Now set up the MLI preconditioner and specify any parameters */
	HYPRE_EuclidCreate(MPI_COMM_WORLD, &precond);

	HYPRE_EuclidSetLevel(precond, 1);

	HYPRE_PCGSetPrecond(solver,
			(HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
			(HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
			precond);

      /* Set some parameters (See Reference Manual for more parameters) */
      //HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
      //HYPRE_ParaSailsSetFilter(precond, sai_filter);
      //HYPRE_ParaSailsSetSym(precond, sai_sym);
      //HYPRE_ParaSailsSetLogging(precond, 3);

      /* Set the PCG preconditioner */
      //HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
      //                    (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup, precond);

	hypre_EndTiming(time_index);
	hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
	hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();

        time_index = hypre_InitializeTiming("PCG Solve");
        hypre_BeginTiming(time_index);
      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

       hypre_EndTiming(time_index);
       hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();


      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      HYPRE_ParCSRPCGDestroy(solver);
	HYPRE_EuclidDestroy(precond);

   }
   /* GMRES */
   else if (solver_id == 54)
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      int    num_iterations;
      double final_res_norm;

      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_GMRESSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_GMRESSetKDim(solver, 30);
      HYPRE_GMRESSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_GMRESSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_GMRESSetLogging(solver, 1); /* needed to get run info later */

      HYPRE_ParCSRGMRESSetup(solver, parcsr_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      time_index = hypre_InitializeTiming("GMRES Solve");
      hypre_BeginTiming(time_index);

      /* Now setup and solve! */
      HYPRE_ParCSRGMRESSolve(solver, parcsr_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();


      /* Run info - needed logging turned on */
      HYPRE_GMRESGetNumIterations(solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      HYPRE_ParCSRGMRESDestroy(solver);
   }
   else if ((solver_id > 55) && (solver_id < 59))
   {
      time_index = hypre_InitializeTiming("GMRES Setup");
      hypre_BeginTiming(time_index);

      int    num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_GMRESSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_GMRESSetKDim(solver, 30);
      HYPRE_GMRESSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_GMRESSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_GMRESSetLogging(solver, 1); /* needed to get run info later */

	if (solver_id == 56)
	{
	// AMG Precond
	      /* Now set up the AMG preconditioner and specify any parameters */
	      printf("AMG Preconditioner\n");
	      HYPRE_BoomerAMGCreate(&precond);
	      HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
	      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
	      HYPRE_BoomerAMGSetOldDefault(precond); 
	      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
	      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
	      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
	      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

	      /* Set the PCG preconditioner */
	      HYPRE_GMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
			          (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

	}
	else if (solver_id == 57)
	{
	// ParaSails Precond
	      int      sai_max_levels = 1;
	      double   sai_threshold = 0.1;
	      double   sai_filter = 0.05;
	      int      sai_sym = 1;
	      printf("ParaSail Preconditioner\n");

	      HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);

	      /* Set some parameters (See Reference Manual for more parameters) */
	      HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
	      HYPRE_ParaSailsSetFilter(precond, sai_filter);
	      HYPRE_ParaSailsSetSym(precond, sai_sym);
	      HYPRE_ParaSailsSetLogging(precond, 3);

	      /* Set the PCG preconditioner */
	      HYPRE_GMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSolve,
		                  (HYPRE_PtrToSolverFcn) HYPRE_ParaSailsSetup, precond);

	}
	else if (solver_id == 58)
	{
	// Euclid Precond
	        printf("Euclid Preconditioner\n");
		HYPRE_EuclidCreate(MPI_COMM_WORLD, &precond);

		HYPRE_EuclidSetLevel(precond, 1);

		HYPRE_GMRESSetPrecond(solver,
				(HYPRE_PtrToSolverFcn) HYPRE_EuclidSolve,
				(HYPRE_PtrToSolverFcn) HYPRE_EuclidSetup,
				precond);
	}

      HYPRE_ParCSRGMRESSetup(solver, parcsr_A, par_b, par_x);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

	time_index = hypre_InitializeTiming("GMRES Solve");
        hypre_BeginTiming(time_index);

      /* Now setup and solve! */
      HYPRE_ParCSRGMRESSolve(solver, parcsr_A, par_b, par_x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();


      /* Run info - needed logging turned on */
      HYPRE_GMRESGetNumIterations(solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      HYPRE_ParCSRGMRESDestroy(solver);

       if (solver_id == 56)
       {
	  HYPRE_BoomerAMGDestroy(precond);
       }
       else if (solver_id == 57)
       {
          HYPRE_ParaSailsDestroy(precond);
       }
       else if (solver_id == 58)
       {
          HYPRE_EuclidDestroy(precond);
       }

   }
   /* Flexible GMRES */
   else if (solver_id == 60)
   {
      int    num_iterations;
      double final_res_norm;
      int    restart = 30;


      /* Create solver */
      HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_FlexGMRESSetKDim(solver, restart);
      HYPRE_FlexGMRESSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_FlexGMRESSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_FlexGMRESSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_FlexGMRESSetLogging(solver, 1); /* needed to get run info later */


      /* Now setup and solve! */
      HYPRE_ParCSRFlexGMRESSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRFlexGMRESSolve(solver, parcsr_A, par_b, par_x);

      /* Run info - needed logging turned on */
      HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      HYPRE_ParCSRFlexGMRESDestroy(solver);

   }
   /* Flexible GMRES with  AMG Preconditioner */
   else if (solver_id == 61)
   {
      int    num_iterations;
      double final_res_norm;
      int    restart = 30;


      /* Create solver */
      HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_FlexGMRESSetKDim(solver, restart);
      HYPRE_FlexGMRESSetMaxIter(solver, 1000); /* max iterations */
      HYPRE_FlexGMRESSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_FlexGMRESSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_FlexGMRESSetLogging(solver, 1); /* needed to get run info later */


      /* Now set up the AMG preconditioner and specify any parameters */
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetOldDefault(precond);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

      //HYPRE_BoomerAMGSetAggNumLevels(precond, 4);

      /* Set the FlexGMRES preconditioner */
      HYPRE_FlexGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                          (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

      /* Now setup and solve! */
      HYPRE_ParCSRFlexGMRESSetup(solver, parcsr_A, par_b, par_x);
      HYPRE_ParCSRFlexGMRESSolve(solver, parcsr_A, par_b, par_x);

      /* Run info - needed logging turned on */
      HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      HYPRE_ParCSRFlexGMRESDestroy(solver);
      HYPRE_BoomerAMGDestroy(precond);

   }
   else
   {
      if (myid ==0) printf("Invalid solver id specified.\n");
   }

    
   /* Save the solution to file */
   HYPRE_ParVectorPrint(par_x, "solution.0");

    
   /* Clean up */
   HYPRE_IJMatrixDestroy(ij_A);
   HYPRE_IJVectorDestroy(ij_b);
   HYPRE_IJVectorDestroy(ij_x);


   /* Finalize MPI*/
   if (myid == 0)
   {
       printf("\n Linear system correctly solved.\n");
       clock_t end = clock();
       double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
       printf("Time spent on HYPRE resolution %f", time_spent);
   }
   MPI_Finalize();

    
   return(0);
}
