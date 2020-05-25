#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include "mat.h"

#define M 3   // match
#define MM -3 // mismatch
#define W 2   // linear gap penalty

/*
Matrix generation and filling

Input:
string1 with length m
string2 with length n

Algorithm:
  two sequences A and B of lenths n and m

1. determine the substitution matrix and the gap penalty scheme
    - s(a,b) similarity score of the elements that consituted the two sequences
    - Wk the penalty of a gap that has length k
2. construct a scoring matrix H and initialize its first row and first column (as 0)
    - size of the scoring matrix is (n+1) * (m+1)
    - Hk0 = H0l = 0 for 0 <= k <= n and 0 <= l <= m
3. fill the scoring matrix using the eqn below

4. traceback
  - starting at the highest score in the scoring matrix H and 
    ending at a matrix cell that has a score of 0, traceback
    based on the source of each score recursively to generate the best
    local alignment

Traceback:
  begin with the highest score, end when 0 is encountered



4. traceback
  - starting at the highest score in the scoring matrix H and 
    ending at a matrix cell that has a score of 0, traceback
    based on the source of each score recursively to generate the best
    local alignment

*/


// generate random sequence of n length
void rand_seq(int n, char seq[]){
  int i;
  for(i = 0; i < n; i++){
    int base = rand() % 4; // random number between 0-3
    switch(base){
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


// calculate similarity score
int sim_score(char a, char b){
  int score;
  if (a == b){
    score = M;
  }
  else {
    score = MM;
  }
  return score;
}


// filling in the scoring matrix
void fill(Matrix h, int seqA_len, char seqA[], int seqB_len, char seqB[]){

	for(int i = 1; i < h.height; i++) {
	  for(int j = 1; j < h.width; j++) {
      int id = i*h.width + j;

      int score = 0;
      h.elements[id] = score;

    }
  }
}


// traceback
void traceback(Matrix h){

}


// input output function to visualzie scoring matrix
void io_score(std::string file, Matrix h, int seqA_len, char seqA[], int seqB_len, char seqB[])
{
	std::ofstream myfile_tsN; 
	myfile_tsN.open(file);

  // print seqA
  myfile_tsN << '\t' << '\t';
  for (int i = 0; i < seqA_len; i++)
     myfile_tsN << seqA[i] << '\t';
  myfile_tsN << std::endl;

  // print vertical seqB with substitution matrix
	for(int i = 0; i < h.height; i++) {
    if (i == 0) {
      myfile_tsN << '\t';
    } 
    else {
      myfile_tsN << seqB[i-1] << '\t';
    }
	  for(int j = 0; j < h.width; j++)
    {
      myfile_tsN << h.elements[i*h.width + j] << '\t';
    }	
    myfile_tsN << std::endl;
  }

	myfile_tsN.close(); 
}


int main(){

  // generate sequences
  int seqA_len = 10;
  int seqB_len = 8;
  char seqA[seqA_len];
  char seqB[seqB_len];
  rand_seq(seqA_len, seqA);
  rand_seq(seqB_len, seqB);

  // print sequences
  std::cout<<"Sequence A is: ";
  for (int i = 0; i < seqA_len; i++)
    std::cout << seqA[i];
  std::cout<< std::endl;

  std::cout<<"Sequence B is: ";
  for (int i = 0; i < seqB_len; i++)
    std::cout << seqB[i];
  std::cout<< std::endl;

  // initialize cpu matrix and apply initial scores of 0
  Matrix h(seqA_len + 1, seqB_len + 1);
	for( int i = 0; i < h.height; i++){
		for( int j = 0; j < h.width; j++) {
          int id = i*h.height + j;
          h.elements[id] = 0;
	  }
	}

  // initialize gpu matrix copy and transfer to device
  int Gpu = 1; 
  Matrix d_h(seqA_len + 1, seqB_len + 1, Gpu);
  d_h.load(h, Gpu); 

  // visualize initial scoring matrix
	std::string s1;
  s1 = "scoreM_init.dat"; 
  io_score(s1, h, seqA_len, seqA, seqB_len, seqB); 

  // fill the scoring matrix
  fill(h, seqA_len, seqA, seqB_len, seqB); 

  // visualize filled scoring matrix
	std::string s2;
  s2 = "scoreM_fill.dat"; 
  io_score(s2, h, seqA_len, seqA, seqB_len, seqB); 

  // traceback
  traceback(h);




  return 0;
}
