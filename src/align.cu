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

/*
Smith-Waterman Alignment of two sequences

Input:
seq1 with length m
seq2 with length n

Algorithm:
1. determine the substitution matrix and the gap penalty scheme
    - s(a,b) similarity score of the elements that consituted the two sequences
    - Wk the penalty of a gap that has length k
2. construct a scoring matrix H and initialize first row and column as 0
    - size of the scoring matrix is (n+1) * (m+1)
    - Hk0 = H0l = 0 for 0 <= k <= n and 0 <= l <= m
3. fill the scoring and direction matrix
    - score each left, above, left-above-diagonal cell depending on the gap,
      match, and mismatch penalties, chosing the maximum score
4. traceback
  - starting at the highest score in the scoring matrix H and
    ending at a matrix cell that has a score of 0, traceback
    based to generate the best local alignment
*/

// generate random sequence of n length
void rand_seq(int n, char seq[]) {
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
  seq[i] = '\0';
}

// filling in the scoring matrix
void fill(Matrix h, Matrix d, char seqA[], char seqB[], int seqA_len,
          int seqB_len) {

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

// traceback: starting at the highest score and ending at the last score get the
// sequence
void traceback(Matrix h, Matrix d, char seqA[], char seqB[],
               std::vector<char> &seqA_aligned,
               std::vector<char> &seqB_aligned) {

  int max_score = 0;
  int max_i;
  int max_j;

  // locating maximum score
  for (int i = 1; i < h.height; i++) {
    for (int j = 1; j < h.width; j++) {
      int id = i * h.width + j;
      if (h.elements[id] > max_score) {
        max_score = h.elements[id];
        max_i = i;
        max_j = j;
      }
    }
  }
  std::cout << "Max score: " << max_score << std::endl;

  // traceback algorithm from maximum score to 0
  while (max_i > 0 && max_j > 0) {

    int id = max_i * h.width + max_j;
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
      std::cout << "Error: Invalid Direction" << std::endl;
      break;
    }
  }
}

// input output function to visualzie matrix
void io_score(std::string file, Matrix h, char seqA[], char seqB[],
              int seqA_len, int seqB_len) {
  std::ofstream myfile_tsN;
  myfile_tsN.open(file);

  // print seqA
  myfile_tsN << '\t' << '\t';
  for (int i = 0; i < seqA_len; i++)
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
  int seqA_len = 12;
  int seqB_len = 12;
  char seqA[seqA_len];
  char seqB[seqB_len];
  rand_seq(seqA_len, seqA);
  rand_seq(seqB_len, seqB);

  // print sequences
  std::cout << "Seq A with length " << seqA_len << " is: ";
  for (int i = 0; i <= seqA_len; i++)
    std::cout << seqA[i];
  std::cout << std::endl;

  std::cout << "Seq B with length " << seqB_len << " is: ";
  for (int i = 0; i <= seqB_len; i++)
    std::cout << seqB[i];
  std::cout << std::endl;

  // initialize cpu matrix and apply initial scores of 0
  Matrix h(seqA_len + 1, seqB_len + 1); // score matrix
  Matrix d(seqA_len + 1, seqB_len + 1); // direction matrix

  for (int i = 0; i < h.height; i++) {
    for (int j = 0; j < h.width; j++) {
      int id = i * h.height + j;
      h.elements[id] = 0;
      d.elements[id] = 0;
    }
  }

  // initialize gpu matrix copy and transfer to device
//  int Gpu = 1;
//  Matrix d_h(seqA_len + 1, seqB_len + 1, Gpu);
//  Matrix d_d(seqA_len + 1, seqB_len + 1, Gpu);
//  d_h.load(h, Gpu);
//  d_d.load(h, Gpu);

  // visualize initial scoring matrix
  std::string s1;
  s1 = "init.dat";
  io_score(s1, h, seqA, seqB, seqA_len, seqB_len);

  // fill the scoring and direction matrix
  fill(h, d, seqA, seqB, seqA_len, seqB_len);

  // visualize filled direction and scoring matrix
  std::string s2;
  std::string s3;
  s2 = "score.dat";
  s3 = "direction.dat";
  io_score(s2, h, seqA, seqB, seqA_len, seqB_len);
  io_score(s3, d, seqA, seqB, seqA_len, seqB_len);

  // optimal alignments
  std::vector<char> seqA_aligned;
  std::vector<char> seqB_aligned;

  // traceback
  traceback(h, d, seqA, seqB, seqA_aligned, seqB_aligned);

  // visualize aligned sequences
  int align_len = seqA_aligned.size();
  std::cout << std::endl;
  std::cout << "Aligned sequences A and B: " << std::endl;
  for (int i = 0; i < align_len + 1; ++i) {
    std::cout << seqA_aligned[align_len - i];
  }
  std::cout << std::endl;

  for (int i = 0; i < align_len + 1; ++i) {
    std::cout << seqB_aligned[align_len - i];
  }
  std::cout << std::endl;

  return 0;
}
