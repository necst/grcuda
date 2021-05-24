/**
 * compile: gcc -o jacobi_openmp -fopenmp jacobi_openmp.c -lm
 * reference: https://people.sc.fsu.edu/~jburkardt/c_src/jacobi_openmp/jacobi_openmp.c
 *
 * */

# include <math.h>
# include <stdio.h>
# include <stdlib.h>

int main ( );

/******************************************************************************/

int main ( )

/******************************************************************************/
/*
 *   Purpose:
 *
 *       MAIN is the main program for JACOBI_OPENMP.
 *
 *         Discussion:
 *
 *             JACOBI_OPENMP carries out a Jacobi iteration with OpenMP.
 *
 *               Licensing:
 *
 *                   This code is distributed under the GNU LGPL license. 
 *
 *                     Modified:
 *
 *                         31 January 2017
 *
 *                           Author:
 *
 *                               John Burkardt
 *                               */
{
  double *b;
  double d;
  int i;
  int it;
  int m;
  int n;
  double r;
  double t;
  double *x;
  double *xnew;

  m = 5000;
  n = 50000;

  b = ( double * ) malloc ( n * sizeof ( double ) );
  x = ( double * ) malloc ( n * sizeof ( double ) );
  xnew = ( double * ) malloc ( n * sizeof ( double ) );

  printf ( "\n" );
  printf ( "JACOBI_OPENMP:\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "  Jacobi iteration to solve A*x=b.\n" );
  printf ( "\n" );
  printf ( "  Number of variables  N = %d\n", n );
  printf ( "  Number of iterations M = %d\n", m );

  printf ( "\n" );
  printf ( "  IT     l2(dX)    l2(resid)\n" );
  printf ( "\n" );

# pragma omp parallel private ( i )
  {
/*
 *   Set up the right hand side.
 *   */
# pragma omp for
    for ( i = 0; i < n; i++ )
    {
      b[i] = 0.0;
    }

    b[n-1] = ( double ) ( n + 1 );
/*
 *   Initialize the solution estimate to 0.
 *     Exact solution is (1,2,3,...,N).
 *     */
# pragma omp for
    for ( i = 0; i < n; i++ )
    {
      x[i] = 0.0;
    }

  }
/*
 *   Iterate M times.
 *   */
  for ( it = 0; it < m; it++ )
  {
# pragma omp parallel private ( i, t )
    {
/*
 *   Jacobi update.
 *   */
# pragma omp for
      for ( i = 0; i < n; i++ )
      {
        xnew[i] = b[i];
        if ( 0 < i )
        {
          xnew[i] = xnew[i] + x[i-1];
        }
        if ( i < n - 1 )
        {
          xnew[i] = xnew[i] + x[i+1];
        }
        xnew[i] = xnew[i] / 2.0;
      }
/*
 *   Difference.
 *   */
      d = 0.0;
# pragma omp for reduction ( + : d )
      for ( i = 0; i < n; i++ )
      {
        d = d + pow ( x[i] - xnew[i], 2 );
      }
/*
 *   Overwrite old solution.
 *   */
# pragma omp for
      for ( i = 0; i < n; i++ )
      {
        x[i] = xnew[i];
      }
/*
 *   Residual.
 *   */
      r = 0.0;
# pragma omp for reduction ( + : r )
      for ( i = 0; i < n; i++ )
      {
        t = b[i] - 2.0 * x[i];
        if ( 0 < i )
        {
          t = t + x[i-1];
        }
        if ( i < n - 1 )
        {
          t = t + x[i+1];
        }
        r = r + t * t;
      }

# pragma omp master
      {
        if ( it < 10 || m - 10 < it )
        {
          printf ( "  %8d  %14.6g  %14.6g\n", it, sqrt ( d ), sqrt ( r ) );
        }
        if ( it == 9 )
        {
          printf ( "  Omitting intermediate results.\n" );
        }
      }

    }

  }
/*
 *   Write part of final estimate.
 *   */
  printf ( "\n" );
  printf ( "  Part of final solution estimate:\n" );
  printf ( "\n" );
  for ( i = 0; i < 10; i++ )
  {
    printf ( "  %8d  %14.6g\n", i, x[i] );
  }
  printf ( "...\n" );
  for ( i = n - 11; i < n; i++ )
  {
    printf ( "  %8d  %14.6g\n", i, x[i] );
  }
/*
 *   Free memory.
 *   */
  free ( b );
  free ( x );
  free ( xnew );
/*
 *   Terminate.
 *   */
  printf ( "\n" );
  printf ( "JACOBI_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}

