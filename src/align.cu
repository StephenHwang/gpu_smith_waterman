#include "mat.h"
#include <chrono>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#define M 3      // match
#define MM -3    // mismatch
#define W -2     // gap score
#define A_LEN 16 // 16, 32, 64, 256, 1024, 2048, 8192  len of sequence A
#define B_LEN 16 // 16, 32, 64, 256, 1024, 2048, 8192 len of sequence B
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Forward declarations of scoring kernel
__global__ void fill_gpu(Matrix h, Matrix d, char seqA[], char seqB[],
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

__global__ void fill_gpu(Matrix h, Matrix d, char seqA[], char seqB[],
                         const int *k, int max_id_val[]) {

  // scores
  int max_score = 0;
  int direction = 0;
  int tmp_score;
  int sim_score;

  // row and column index depending on anti-diagonal
  int i = threadIdx.x + 1 + blockDim.x * blockIdx.x;
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

  // similarity score for diagonal cell
  char baseA = seqA[j - 1];
  char baseB = seqB[i - 1];
  if (baseA == baseB) {
    sim_score = M;
  } else {
    sim_score = MM;
  }

  // diagonal cell (preferred)
  tmp_score = h.elements[diag_id] + sim_score;
  if (tmp_score >= max_score) {
    max_score = tmp_score;
    direction = 3;
  }

  // assign scores and direction
  h.elements[id] = max_score;
  d.elements[id] = direction;

  // save max score and position
  if (max_score > max_id_val[1]) {
    max_id_val[0] = id;
    max_id_val[1] = max_score;
  }
}

// cpu finding index location of maximum score
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
  std::cout << "   ";
  for (int i = 0; i < align_len + 1; ++i) {
    std::cout << seqA_aligned[align_len - i];
  }
  std::cout << std::endl;

  std::cout << "   ";
  for (int i = 0; i < align_len + 1; ++i) {
    std::cout << seqB_aligned[align_len - i];
  }
  std::cout << std::endl << std::endl;
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

void smith_water_cpu(Matrix h, Matrix d, char seqA[], char seqB[]) {

  // populate scoring and direction matrix and find id of max score
  int max_id = fill_cpu(h, d, seqA, seqB);

  // traceback
  std::vector<char> seqA_aligned;
  std::vector<char> seqB_aligned;
  traceback(d, max_id, seqA, seqB, seqA_aligned, seqB_aligned);

  // print aligned sequences
  io_seq(seqA_aligned, seqB_aligned);

  std::cout << std::endl;
  std::cout << "CPU result: " << std::endl;

  // print cpu populated direction and scoring matrix
  io_score(std::string("score.dat"), h, seqA, seqB);
  io_score(std::string("direction.dat"), d, seqA, seqB);
}

void smith_water_gpu(Matrix h, Matrix d, char seqA[], char seqB[]) {

  std::cout << "GPU result: " << std::endl;

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

  // max id and value
  int *d_max_id_val;                   // create pointers and device
  std::vector<int> h_max_id_val(2, 0); // allocate and initialize mem on host
  cudaMalloc(&d_max_id_val, 2 * sizeof(int)); // allocate memory on GPU
  cudaMemcpy(d_max_id_val, h_max_id_val.data(), 2 * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // loop over diagonals of the matrix
  for (int i = 1; i <= ((A_LEN + 1) + (B_LEN + 1) - 1); i++) {
    int col_idx = max(0, (i - (B_LEN + 1)));
    int diag_len = min(i, ((A_LEN + 1) - col_idx));

    // launch the kernel: one block by length of diagonal
    cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);

    int blks = 16;
    dim3 dimBlock(diag_len / blks);
    dim3 dimGrid(blks);

    fill_gpu<<<dimGrid, dimBlock>>>(d_h, d_d, d_seqA, d_seqB, d_i,
                                    d_max_id_val);
    cudaDeviceSynchronize();
  }

  // copy data back
  size_t size = (A_LEN + 1) * (B_LEN + 1) * sizeof(float);
  cudaMemcpy(d.elements, d_d.elements, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h.elements, d_h.elements, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_max_id_val.data(), d_max_id_val, 2 * sizeof(int),
             cudaMemcpyDeviceToHost);

  //  std::cout << "   Max score of " << h_max_id_val[1] << " at " <<
  //  max_id_val[0]
  //            << std::endl;

  // traceback
  int max_id = h_max_id_val[0];
  std::vector<char> seqA_aligned;
  std::vector<char> seqB_aligned;
  traceback(d, max_id, seqA, seqB, seqA_aligned, seqB_aligned);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "   GPU time = " << time << " ms" << std::endl;

  // visualize output
  // io_seq(seqA_aligned, seqB_aligned);
  // io_score(std::string("score_gpu.dat"), h, seqA, seqB);
  // io_score(std::string("direction_gpu.dat"), d, seqA, seqB);

  // deallocate memory
  d_h.gpu_deallocate();
  d_d.gpu_deallocate();
  cudaFree(d_seqA);
  cudaFree(d_seqB);
  cudaFree(d_i);
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

  // initialize scoring and direction matrices
  Matrix scr_cpu(A_LEN + 1, B_LEN + 1); // cpu score matrix
  Matrix dir_cpu(A_LEN + 1, B_LEN + 1); // cpu direction
  Matrix scr_gpu(A_LEN + 1, B_LEN + 1); // gpu score matrix
  Matrix dir_gpu(A_LEN + 1, B_LEN + 1); // gpu direction matrix

  // apply initial condition of 0
  for (int i = 0; i < scr_cpu.height; i++) {
    for (int j = 0; j < scr_cpu.width; j++) {
      int id = i * scr_cpu.width + j;
      scr_cpu.elements[id] = 0;
      dir_cpu.elements[id] = 0;
      scr_gpu.elements[id] = 0;
      dir_gpu.elements[id] = 0;
    }
  }

  // visualize initial scoring matrix
  io_score(std::string("init.dat"), scr_cpu, seqA, seqB);

  // CPU
  auto start_cpu = std::chrono::steady_clock::now();
  smith_water_cpu(scr_cpu, dir_cpu, seqA, seqB); // call CPU smith water
  auto end_cpu = std::chrono::steady_clock::now();
  auto diff = end_cpu - start_cpu;
  std::cout << "   CPU time = "
            << std::chrono::duration<double, std::milli>(diff).count() << " ms"
            << std::endl;
  std::cout << std::endl;

  // GPU
  smith_water_gpu(scr_gpu, dir_gpu, seqA, seqB); // call GPU smith water

  // deallocate memory
  scr_cpu.cpu_deallocate();
  dir_cpu.cpu_deallocate();
  scr_gpu.cpu_deallocate();
  dir_gpu.cpu_deallocate();

  return 0;
}
