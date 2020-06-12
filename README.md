# Smith-Waterman Local Sequence Alignment

A CUDA C++/C++  Smith-Waterman local alignment algorithm for parallel computation across Nvidia GPUs.

## Sequence Alignment
The Smith-Waterman local alignment algorithm is a dynamic programming approach to identify regions of similarity across a pair of sequences. Due to its O(mn) complexity, Smith-Waterman suffers with large sequence length. Here I alter the original Smith-Waterman algorithm and parallelize computation of elements within diagonals of the scoring matrix to increase performance.

