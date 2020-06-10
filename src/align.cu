#include "mat.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define M 3   // match
#define MM -3 // mismatch
#define W -2  // gap score
#define A_LEN 32 // length of sequence A
#define B_LEN 32 // length of sequence B
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

// Forward declarations of kernels and functions
void seq_gen(int n, char seq[]);
void fill_cpu(Matrix h, Matrix d, char seqA[], char seqB[]);
__global__ void naive_fill_gpu(Matrix h, Matrix d, char seqA[], char seqB[], const int *k);
int max_score_cpu(Matrix h);
void traceback(Matrix d, int max_id, char seqA[], char seqB[],
               std::vector<char> &seqA_aligned,
               std::vector<char> &seqB_aligned);
void io_score(std::string file, Matrix h, char seqA[], char seqB[]);
void s_seq(std::vector<char> &seqA_aligned, std::vector<char> &seqB_aligned);


// generate random sequence of n length
void seq_gen(int n, char seq[]) {
  int i;
  for (i = 0; i < n; i++) {
    int base = rand() % 4;
    switch (base) {
    case 0:
      seq[i] = 'A';
      break;
    case 1:
      seq[i] = 'T';
      break;
    case 2:
      seq[i] = 'C';
      break;
    case 3:
      seq[i] = 'G';
      break;
    }
  }
}

// filling in the scoring matrix
void fill_cpu(Matrix h, Matrix d, char seqA[], char seqB[]) {
  for (int i = 1; i < h.height; i++) {
    for (int j = 1; j < h.width; j++) {

      // scores
      int max_score = 0;
      int direction = 0;
      int tmp_score;
      int sim_score;

      // comparison positions
      int id = i * h.width + j;            // current cell
      int abov_id = (i - 1) * h.width + j; // above cell, 1
      int left_id = i * h.width + (j - 1); // left cell, 2
      int diag_id = (i - 1) * h.width + (j - 1); // upper-left diagonal cell, 3

      // above cell
      tmp_score = h.elements[abov_id] + W;
      if (tmp_score > max_score) {
        max_score = tmp_score;
        direction = 1;
      }

      // left cell
      tmp_score = h.elements[left_id] + W;
      if (tmp_score > max_score) {
        max_score = tmp_score;
        direction = 2;
      }

      // diagonal cell (preferred)
      char baseA = seqA[j - 1];
      char baseB = seqB[i - 1];
      if (baseA == baseB) {
        sim_score = M;
      } else {
        sim_score = MM;
      }

      tmp_score = h.elements[diag_id] + sim_score;
      if (tmp_score >= max_score) {
        max_score = tmp_score;
        direction = 3;
      }

      // assign scores and direction
      h.elements[id] = max_score;
      d.elements[id] = direction;
    }
  }
}


__global__ void naive_fill_gpu(Matrix h, Matrix d, char seqA[], char seqB[], const int *k) {

  // scores
  int max_score = 0;
  int direction = 0;
  int tmp_score;
  int sim_score;

  // row and column index depending on anti-diagonal
  int i = threadIdx.x + 1;
  if (*k > A_LEN + 1) {
    i += (*k - A_LEN);
  }
  int j = ((*k)-i)+1;

  // comparison positions
  int id = i * h.width + j;
  int abov_id = (i - 1) * h.width + j; // above cell, 1
  int left_id = i * h.width + (j - 1); // left cell, 2
  int diag_id = (i - 1) * h.width + (j - 1); // upper-left diagonal cell, 3

// above cell
  tmp_score = h.elements[abov_id] + W;
  if (tmp_score > max_score) {
    max_score = tmp_score;
    direction = 1;
  }

  // left cell
  tmp_score = h.elements[left_id] + W;
  if (tmp_score > max_score) {
    max_score = tmp_score;
    direction = 2;
  }

  // diagonal cell (preferred)
  char baseA = seqA[j - 1];
  char baseB = seqB[i - 1];
  if (baseA == baseB) {
    sim_score = M;
  } else {
    sim_score = MM;
  }

  tmp_score = h.elements[diag_id] + sim_score;
  if (tmp_score >= max_score) {
    max_score = tmp_score;
    direction = 3;
  }

  // assign scores and direction
  h.elements[id] = max_score;
  d.elements[id] = direction;
}


// find index location of maximum score
int max_score_cpu(Matrix h) {

  int max_score = 0;
  int max_id;

  // locating maximum score
  for (int i = 1; i < h.height; i++) {
    for (int j = 1; j < h.width; j++) {
      int id = i * h.width + j;
      if (h.elements[id] > max_score) {
        max_score = h.elements[id];
        max_id = id;
      }
    }
  }
  std::cout << "Max score of " << max_score;
  std::cout << " at id: " << max_id << std::endl;

  return max_id;
}


// traceback: starting at the highest score and ending at a 0 score
void traceback(Matrix d, int max_id, char seqA[], char seqB[],
               std::vector<char> &seqA_aligned,
               std::vector<char> &seqB_aligned) {

  int max_i = max_id / d.width;
  int max_j = max_id % d.width ;

  // traceback algorithm from maximum score to 0
  while (max_i > 0 && max_j > 0) {

    int id = max_i * d.width + max_j;
    int dir = d.elements[id];

    switch (dir) {
    case 1:
      --max_i;
      seqA_aligned.push_back('-');
      seqB_aligned.push_back(seqB[max_i]);
      break;
    case 2:
      --max_j;
      seqA_aligned.push_back(seqA[max_j]);
      seqB_aligned.push_back('-');
      break;
    case 3:
      --max_i;
      --max_j;
      seqA_aligned.push_back(seqA[max_j]);
      seqB_aligned.push_back(seqB[max_i]);
      break;
    case 0:
      max_i = -1;
      max_j = -1;
      break;
    }
  }
}


// print aligned sequnces
void io_seq(std::vector<char> &seqA_aligned, std::vector<char> &seqB_aligned) {

  std::cout << "Aligned sub-sequences of A and B: " << std::endl;
  int align_len = seqA_aligned.size();
  for (int i = 0; i < align_len + 1; ++i) {
    std::cout << seqA_aligned[align_len - i];
  }
  std::cout << std::endl;

  for (int i = 0; i < align_len + 1; ++i) {
    std::cout << seqB_aligned[align_len - i];
  }
  std::cout << std::endl;
}


// input output function to visualize matrix
void io_score(std::string file, Matrix h, char seqA[], char seqB[]) {
  std::ofstream myfile_tsN;
  myfile_tsN.open(file);

  // print seqA
  myfile_tsN << '\t' << '\t';
  for (int i = 0; i < A_LEN; i++)
    myfile_tsN << seqA[i] << '\t';
  myfile_tsN << std::endl;

  // print vertical seqB on left of matrix 
  for (int i = 0; i < h.height; i++) {
    if (i == 0) {
      myfile_tsN << '\t';
    } else {
      myfile_tsN << seqB[i - 1] << '\t';
    }
    for (int j = 0; j < h.width; j++) {
      myfile_tsN << h.elements[i * h.width + j] << '\t';
    }
    myfile_tsN << std::endl;
  }
  myfile_tsN.close();
}


int main() {

  // generate sequences
  char seqA[A_LEN];
  char seqB[B_LEN];
  seq_gen(A_LEN, seqA);
  seq_gen(B_LEN, seqB);

  // print sequences
  std::cout << "Seq A with length " << A_LEN << " is: ";
  for (int i = 0; i < A_LEN; i++)
    std::cout << seqA[i];
  std::cout << std::endl;

  std::cout << "Seq B with length " << B_LEN << " is: ";
  for (int i = 0; i < B_LEN; i++)
    std::cout << seqB[i];
  std::cout << std::endl;
  std::cout << std::endl;

  // initialize cpu matrices
  Matrix h1(A_LEN + 1, B_LEN + 1); // score matrix
  Matrix d1(A_LEN + 1, B_LEN + 1); // direction matrix

  // initialize gpu matrices
  Matrix h2(A_LEN + 1, B_LEN + 1); // score matrix
  Matrix d2(A_LEN + 1, B_LEN + 1); // direction matrix

  // apply initial condition of 0
  for (int i = 0; i < h1.height; i++) {
    for (int j = 0; j < h1.width; j++) {
      int id = i * h1.width + j;
      h1.elements[id] = 0;
      d1.elements[id] = 0;
      h2.elements[id] = 0;
      d2.elements[id] = 0;
    }
  }

  // initialize matrices for gpu
  int Gpu = 1;
  Matrix d_h(A_LEN + 1, B_LEN + 1, Gpu);
  Matrix d_d(A_LEN + 1, B_LEN + 1, Gpu);

  // transfer to device
  d_h.load(h2, Gpu);
  d_d.load(d2, Gpu);

  // visualize initial scoring matrix
//  std::string s1;
//  s1 = "init.dat";
//  io_score(s1, h1, seqA, seqB);

/////////////////////////////////////////////////////////////////////////////

  // working cpu
  fill_cpu(h1, d1, seqA, seqB);

  // cpu traceback
  std::cout << "CPU result: " << std::endl;
  std::vector<char> seqA_aligned1;
  std::vector<char> seqB_aligned1;
  int max_id1 = max_score_cpu(h1);
  traceback(d1, max_id1, seqA, seqB, seqA_aligned1, seqB_aligned1);

  // visualize aligned sequences
  io_seq(seqA_aligned1, seqB_aligned1);
  std::cout << std::endl;

  // print cpu populated direction and scoring matrix
  std::string s2;
  std::string s3;
  s2 = "score.dat";
  s3 = "direction.dat";
  io_score(s2, h1, seqA, seqB);
  io_score(s3, d1, seqA, seqB);


/////////////////////////////////////////////////////////////////////////////

  // naive GPU
  int *d_i;
  cudaMalloc((void**)&d_i, sizeof(int));

  // transfer sequence data to device
	char *d_seqA, *d_seqB; 
  cudaMalloc(&d_seqA, A_LEN * sizeof(char)); 
  cudaMalloc(&d_seqB, B_LEN * sizeof(char)); 

  cudaMemcpy(d_seqA, seqA, A_LEN * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_seqB, seqB, B_LEN * sizeof(char), cudaMemcpyHostToDevice);

  // looping over diagonals of the matrix
  for (int i = 1; i <= ((A_LEN + 1) + (B_LEN+ 1) - 1); i++){
    int col_idx = max(0, (i - (B_LEN +1))); // column index
    int diag_len = min(i, ((A_LEN + 1) - col_idx)); // length of diagonal

    std::cout << "Index: " << i << ", Column: " << col_idx;
    std::cout << ", with len: " << diag_len << std::endl;

    // launch the kernel: one block with length of diagonal
    cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);
    naive_fill_gpu<<<1, diag_len>>>(d_h, d_d, d_seqA, d_seqB, d_i);
    cudaDeviceSynchronize();
  }

  // gpu traceback
  std::cout << "GPU result: " << std::endl;
  std::vector<char> seqA_aligned2;
  std::vector<char> seqB_aligned2;

  // copy back data
  size_t size = (A_LEN + 1)*(B_LEN + 1)*sizeof(float);
  cudaMemcpy(h2.elements, d_h.elements, size, cudaMemcpyDeviceToHost);  
  cudaMemcpy(d2.elements, d_d.elements, size, cudaMemcpyDeviceToHost);  

  std::string s4;
  std::string s5;
  s4 = "score_gpu.dat";
  s5 = "direction_gpu.dat";
  io_score(s4, h2, seqA, seqB);
  io_score(s5, d2, seqA, seqB);

  int max_id2 = max_score_cpu(h2);
  traceback(d2, max_id2, seqA, seqB, seqA_aligned2, seqB_aligned2);
  io_seq(seqA_aligned2, seqB_aligned2);


/////////////////////////////////////////////////////////////////////////////

  // deallocate cpu and gpu memory
  h1.cpu_deallocate();
  d1.cpu_deallocate();
  h2.cpu_deallocate();
  d2.cpu_deallocate();

  d_h.gpu_deallocate(); 
  d_d.gpu_deallocate();
  cudaFree(d_seqA);
  cudaFree(d_seqB);

  return 0;
}
