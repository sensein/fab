#!/bin/bash                      
#SBATCH --job-name=cvd
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH -t 30:00:00          # walltime = 1 hours and 30 minutes
#SBATCH -N 1                 # one node
#SBATCH -c 10                # two CPU (hyperthreaded) cores
#SBATCH --mem=80G
#SBATCH --partition=gablab
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-A100
#SBATCH -x node[103]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabiocat@mit.edu



#SBATCH --array=1-12

# Define your lists of encoders and decoders
ENCODERS=("encoder1" "encoder2" "encoder3" "encoder4")
DECODERS=("decoder1" "decoder2" "decoder3")

# Calculate the current encoder and decoder based on the array index
ENCODER_INDEX=$(((SLURM_ARRAY_TASK_ID - 1) / ${#DECODERS[@]}))
DECODER_INDEX=$(((SLURM_ARRAY_TASK_ID - 1) % ${#DECODERS[@]}))

# Get the current encoder and decoder based on the calculated indices
ENCODER=${ENCODERS[$ENCODER_INDEX]}
DECODER=${DECODERS[$DECODER_INDEX]}

# Execute commands to run your program here. Here is an example of python.
eval "$(conda shell.bash hook)"
conda activate fab

cd ../train/
python train.py --encoder "$ENCODER" --decoder "$DECODER"
