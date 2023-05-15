#!/bin/bash                      
#SBATCH --job-name=cvd

log_path="../logs"
if [[ ! -e $dir ]]; then
    mkdir -p $log_path
fi

#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH -t 1:00:00          # walltime = 1 hours and 30 minutes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabiocat@mit.edu

# Define your lists of encoders and decoders
ENCODERS=("encoder1" "encoder2" "encoder3" "encoder4")
DECODERS=("decoder1" "decoder2" "decoder3")

# Calculate the array size based on the length of encoders and decoders
ARRAY_SIZE=$(( ${#ENCODERS[@]} * ${#DECODERS[@]} ))

# Set the array variable based on the calculated array size
#SBATCH --array=1-$ARRAY_SIZE

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
python hpt.py --encoder "$ENCODER" --decoder "$DECODER"
python train.py --encoder "$ENCODER" --decoder "$DECODER"
