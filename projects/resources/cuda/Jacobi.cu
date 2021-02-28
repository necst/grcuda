# include <cmath>
# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>
# include <omp.h>


__global__ void Jacobi_update(float *x, float *xnew, int n){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
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
}

__global__ float reduction(float *x, float *xnew, int n, float d){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        d = d + pow ( x[i] - xnew[i], 2 );
    }
    return d;
}

Jacobi(Options &options) : Benchmark(options) {}
void alloc(){
    err = cudaMallocManaged(&x, sizeof(float) * N);
    err = cudaMallocManaged(&b, sizeof(float) * N);
    err = cudaMallocManaged(&xnew, sizeof(float) * N);
}
void init(){
    for ( int i = 0; i < N; i++ )
    {
      b[i] = 0.0;
      x[i] = 0.0;
    }
    d = 0.0;
    b[N-1] = ( float ) ( N + 1 );
}
void reset(){
    for ( int i = 0; i < N; i++ )
    {
      b[i] = 0.0;
      x[i] = 0.0;
      xnew[i] = 0.0;
    }
    d = 0.0

}
void execute_sync(int iter);
void execute_async(int iter);
void execute_cudagraph(int iter);
void execute_cudagraph_manual(int iter);
void execute_cudagraph_single(int iter);
void prefetch(cudaStream_t &s1, cudaStream_t &s2);
std::string print_result(bool short_form = false);



int main ( )

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

  b = new double[n];
  x = new double[n];
  xnew = new double[n];

  timestamp ( );
  cout << "\n";
  cout << "JACOBI_OPENMP:\n";
  cout << "  C/OpenMP version\n";
  cout << "  Jacobi iteration to solve A*x=b.\n";
  cout << "\n";
  cout << "  Number of variables  N = " << n << "\n";
  cout << "  Number of iterations M = " <<  m << "\n";

  cout << "\n";
  cout << "  IT     l2(dX)    l2(resid)\n";
  cout << "\n";

# pragma omp parallel private ( i )
  {
//
//  Set up the right hand side.
//
# pragma omp for
    for ( i = 0; i < n; i++ )
    {
      b[i] = 0.0;
    }

    b[n-1] = ( double ) ( n + 1 );
//
//  Initialize the solution estimate to 0.
//  Exact solution is (1,2,3,...,N).
//
# pragma omp for
    for ( i = 0; i < n; i++ )
    {
      x[i] = 0.0;
    }

  }
//
//  Iterate M times.
//
  for ( it = 0; it < m; it++ )
  {
# pragma omp parallel private ( i, t )
    {
//
//  Jacobi update.
//
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
//
//  Difference.
//
      d = 0.0;
# pragma omp for reduction ( + : d )
      for ( i = 0; i < n; i++ )
      {
        d = d + pow ( x[i] - xnew[i], 2 );
      }
//
//  Overwrite old solution.
//
# pragma omp for
      for ( i = 0; i < n; i++ )
      {
        x[i] = xnew[i];
      }
//
//  Residual.
//
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
          cout << "  " << setw(8) << it
               << "  " << setw(14) << sqrt ( d )
               << "  " << sqrt ( r ) << "\n";
        }
        if ( it == 9 )
        {
          cout << "  Omitting intermediate results.\n";
        }
      }

    }

  }
//
//  Write part of final estimate.
//
  cout << "\n";
  cout << "  Part of final solution estimate:\n";
  cout << "\n";
  for ( i = 0; i < 10; i++ )
  {
    cout << "  " << setw(8) << i
         << "  " << setw(14) << x[i] << "\n";
  }
  cout << "...\n";
  for ( i = n - 11; i < n; i++ )
  {
    cout << "  " << setw(8) << i
         << "  " << setw(14) << x[i] << "\n";
  }
//
//  Free memory.
//
  delete [] b;
  delete [] x;
  delete [] xnew;
//
//  Terminate.
//
  cout << "\n";
  cout << "JACOBI_OPENMP:\n";
  cout << "  Normal end of execution.\n";
  cout << "\n";
  timestamp ( );

  return 0;
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 March 2018
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  std::time_t now;

  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );

  std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

  std::cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}
