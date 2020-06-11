#include "mat.h"
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define M 3        // match
#define MM -3      // mismatch
#define W -2       // gap score
#define A_LEN 32 // length of sequence A
#define B_LEN 32 // length of sequence B
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Forward declarations of scoring kernels
__global__ void fill_gpu_naive(Matrix h, Matrix d, char seqA[], char seqB[],
                               const int *k);

// generate random sequence of length n
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

//void fill_cpu(Matrix h, Matrix d, char seqA[], char seqB[]) {
int fill_cpu(Matrix h, Matrix d, char seqA[], char seqB[]) {

  int full_max_id = 0;
  int full_max_val = 0;

  for (int i = 1; i < h.height; i++) {
    for (int j = 1; j < h.width; j++) {

      // scores
      int max_score = 0;
      int direction = 0;
      int tmp_score;
      int sim_score;

      // comparison positions
      int id = i * h.width + j;                  // current cell
      int abov_id = (i - 1) * h.width + j;       // above cell, 1
      int left_id = i * h.width + (j - 1);       // left cell, 2
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

      if (max_score > full_max_val) {
        full_max_id = id;
        full_max_val = max_score;

      }
    }
  }
  
  std::cout << "Max score of " << full_max_val;
  std::cout << " at id: " << full_max_id << std::endl;
  return full_max_id;
}

__global__ void fill_gpu_naive(Matrix h, Matrix d, char seqA[], char seqB[],
                               const int *k, int max_id_val[]) {

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
  int j = ((*k) - i) + 1;

  // comparison positions
  int id = i * h.width + j;
  int abov_id = (i - 1) * h.width + j;       // above cell, 1
  int left_id = i * h.width + (j - 1);       // left cell, 2
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

  if (max_score > max_id_val[1]) {
    max_id_val[0] = id;
    max_id_val[1] = max_score;
  }
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
  int max_j = max_id % d.width;

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

void smith_water_CPU(Matrix h, Matrix d, char seqA[], char seqB[]) {
  std::cout << "CPU result: " << std::endl;

  // populate scoring and direction matrix and find id of max score
  int max_id = fill_cpu(h, d, seqA, seqB);

  // traceback
  std::vector<char> seqA_aligned;
  std::vector<char> seqB_aligned;
  traceback(d, max_id, seqA, seqB, seqA_aligned, seqB_aligned);

  // print aligned sequences
  io_seq(seqA_aligned, seqB_aligned);

  // print cpu populated direction and scoring matrix
  io_score(std::string("score.dat"), h, seqA, seqB);
  io_score(std::string("direction.dat"), d, seqA, seqB);
}

void smith_water_GPU_naive(Matrix h, Matrix d, char seqA[], char seqB[]) {

  // allocate and transfer sequence data to device
  char *d_seqA, *d_seqB;
  cudaMalloc(&d_seqA, A_LEN * sizeof(char));
  cudaMalloc(&d_seqB, B_LEN * sizeof(char));
  cudaMemcpy(d_seqA, seqA, A_LEN * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_seqB, seqB, B_LEN * sizeof(char), cudaMemcpyHostToDevice);

  // initialize matrices for gpu
  int Gpu = 1;
  Matrix d_h(A_LEN + 1, B_LEN + 1, Gpu);
  Matrix d_d(A_LEN + 1, B_LEN + 1, Gpu);
  d_h.load(h, Gpu);
  d_d.load(d, Gpu);

  // device index
  int *d_i;
  cudaMalloc(&d_i, sizeof(int));

  // timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // max id and value
	int *d_max_id_val;  // create pointers and device
	std::vector<int> h_max_id_val(2, 0); // allocate and initialize mem on host

	cudaMalloc(&d_max_id_val, 2*sizeof(int)); // allocate memory on GPU
	cudaMemcpy(d_max_id_val, h_max_id_val.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

  // loop over diagonals of the matrix
  for (int i = 1; i <= ((A_LEN + 1) + (B_LEN + 1) - 1); i++) {
    int col_idx = max(0, (i - (B_LEN + 1)));
    int diag_len = min(i, ((A_LEN + 1) - col_idx));

    // launch the kernel: one block by length of diagonal
    cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);
    fill_gpu_naive<<<1, diag_len>>>(d_h, d_d, d_seqA, d_seqB, d_i, d_max_id_val);
    cudaDeviceSynchronize();
  }

  // gpu traceback //
  std::cout << "GPU result: " << std::endl;
  std::vector<char> seqA_aligned;
  std::vector<char> seqB_aligned;

  // copy data back
  size_t size = (A_LEN + 1) * (B_LEN + 1) * sizeof(float);
//  cudaMemcpy(h.elements, d_h.elements, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(d.elements, d_d.elements, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_max_id_val.data(), d_max_id_val, 2*sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Max score of " << h_max_id_val[1] << " at " << h_max_id_val[0] << std::endl;

//  int max_id = max_score_cpu(h);
  int max_id = h_max_id_val[0];
  traceback(d, max_id, seqA, seqB, seqA_aligned, seqB_aligned);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "   Naive GPU time = " << time << " ms" << std::endl;

  // visualize output
  io_seq(seqA_aligned, seqB_aligned);
  std::string s4;
  std::string s5;
  s4 = "score_gpu.dat";
  s5 = "direction_gpu.dat";
  io_score(s4, h, seqA, seqB);
  io_score(s5, d, seqA, seqB);

  d_h.gpu_deallocate();
  d_d.gpu_deallocate();
  cudaFree(d_seqA);
  cudaFree(d_seqB);
	cudaFree(d_max_id_val);
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

  // initialize scoring and direction matrices
  Matrix h1(A_LEN + 1, B_LEN + 1); // score matrix for cpu
  Matrix d1(A_LEN + 1, B_LEN + 1); // direction matrix for cpu
  Matrix h2(A_LEN + 1, B_LEN + 1); // score matrix for gpu naive
  Matrix d2(A_LEN + 1, B_LEN + 1); // direction matrix for gpu naive

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

  // visualize initial scoring matrix
  io_score(std::string("init.dat"), h1, seqA, seqB);

  // CPU
  clock_t begin = clock();
  smith_water_CPU(h1, d1, seqA, seqB);
  clock_t end = clock();
  double cpu = double(end - begin) / (CLOCKS_PER_SEC * 12);
  std::cout << "   CPU time = " << cpu * 1000 << " ms" << std::endl;
  std::cout << std::endl;

  // naive GPU
  smith_water_GPU_naive(h2, d2, seqA, seqB);

  // deallocate memory
  h1.cpu_deallocate();
  d1.cpu_deallocate();
  h2.cpu_deallocate();
  d2.cpu_deallocate();

  return 0;
}
