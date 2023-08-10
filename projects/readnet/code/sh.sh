#!/bin/bash                      
#SBATCH --job-name=anon
#SBATCH --output=./logs/%A_%a.out
#SBATCH --error=./logs/%A_%a.err
#SBATCH -t 96:00:00          # walltime = 96 hours (adjust as needed)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabiocat@mit.edu
#SBATCH -N 1
#SBATCH -n 8                     # 8 CPU cores (adjust as needed)
#SBATCH --mem=100GB               # 80GB memory (adjust as needed)
#SBATCH --gres=shard:1 
#SBATCH -x node[100-106,109,110]     # Exclude certain nodes if needed
#SBATCH --constraint=any-A100
#SBATCH -p gablab
#SBATCH --array=1-75%1           # The array will run from 1 to 5, with a step size of 1

# Load necessary modules (if needed)
# module load python

# Execute commands to run your program here. Here is an example of python.
eval "$(conda shell.bash hook)"
conda activate fab

# Define arrays for dataset names and anonymization tool names
dataset_names=("English_ECSC_men_dataset" "kennedy_james_freespeech_dataset" "kennedy_james_sentencerepetition_dataset" "English_ECSC_child_dataset" "English_ECSC_women_dataset")
anonymization_tool_names=("coqui" "mcadams" "freevc")

# Function to get target speaker based on dataset name
get_target_speaker() {
  if [[ $1 == "English_ECSC_women_dataset" ]]; then
    echo "sally_female_adult_aws.wav"
  elif [[ $1 == "English_ECSC_men_dataset" ]]; then
    echo "stephen_male_adult_aws.wav"
  else
    echo "kevin_male_child_aws.wav"
  fi
}

# Calculate the indices for the current combination of dataset and tool
let "dataset_index = ($SLURM_ARRAY_TASK_ID - 1) / (${#anonymization_tool_names[@]} * 5)"
let "tool_index = (($SLURM_ARRAY_TASK_ID - 1) / 5) % ${#anonymization_tool_names[@]}"
let "seed = ($SLURM_ARRAY_TASK_ID - 1) % 5"

dataset_name="${dataset_names[$dataset_index]}"
anonymization_tool_name="${anonymization_tool_names[$tool_index]}"

# Calculate the target speaker based on the dataset name
target_speaker=$(get_target_speaker "$dataset_name")

# Print the current task information
echo "Running voice-anonymization_project.py with dataset_name: $dataset_name, anonymization_tool_name: $anonymization_tool_name, target_speaker_for_anonymization_file: $target_speaker, seed: $seed"

# Print the current task information
python voice-anonymization_project.py --dataset_name "$dataset_name" --anonymization_tool_name "$anonymization_tool_name" --target_speaker_for_anonymization_file "$target_speaker" --seed "$seed"
