## Benchmarks and Utilities

This folder contains the Java, Python and CUDA benchmarks along with the scripts to compute the interconnection matrix within the current architecture. 

| **Benchmark #** 	| **Benchmark Name** 	| **Extended Description**                                                                                                                                                          	|
|-----------------	|--------------------	|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| B1M             	| VEC                	| _Compute the sum of difference of squares of 2 vectors, using multiple GrCUDA kernels. Parallelize the computation on multiple GPUs, by computing a chunk of the output on each._ 	|
| B5M             	| B&S                	| _Black-Scholes equation benchmark, executed concurrently on different input vectors._                                                                                             	|
| B6M             	| ML                 	| _Compute an ensemble of Categorical Naive Bayes and Ridge Regression classifiers. Predictions are aggregated averaging the class scores after softmax normalizatio._              	|
| B9M             	| CG                 	| _Compute the conjugate gradient algorithm on a dense symmetric matrix. The matrix-vector multiplications are row-partitioned to scale across multiple GPUs._                      	|
| B11M            	| MUL                	| _Dense matrix-vector multiplication, partitioning the matrix in blocks of rows._                                                                                                  	|

