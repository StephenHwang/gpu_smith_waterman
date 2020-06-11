#!/bin/bash
# smith waterman sequence alignment
#SBATCH --job-name=sw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sjhwang@ucsc.edu
#SBATCH --output=matvec_%j.out
#SBATCH --error=matvec_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=am-148-s20
#SBATCH --qos=am-148-s20
#SBATCH --account=am-148-s20 

module load cuda10.0
/home/sjhwang/finalProject/seq_assembly/src/align.exe
