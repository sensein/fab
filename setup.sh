#!/bin/bash

#################### config #########################
env_name="fab"
tools_folder="tools" 
conda_sh="/Users/fabiocat/miniconda3/etc/profile.d/conda.sh"
#################### config #########################

######################################## start #############################################

echo "Hello, $env_name!"

if conda info --envs | grep -q "/$env_name$"; then
  echo "Environment '$env_name' already exists"
else
  echo "Environment '$env_name' does not exist"
  conda create -n $env_name python=3.8 -y
fi

source $conda_sh
conda activate $env_name
echo "Active Conda environment: $CONDA_DEFAULT_ENV"

if [ -d "$tools_folder" ]; then
  echo "Folder '$tools_folder' already exists"
else
  echo "Folder '$tools_folder' does not exist"
  mkdir $tools_folder
fi

cd $tools_folder

#################### pyannote-audio #########################
repo_url="https://github.com/fabiocat93/pyannote-audio.git"
repo_dir="pyannote-audio"

if [ -d "$repo_dir" ]; then
  # Repository exists, pull updates
  echo "Repository '$repo_dir' exists, pulling updates..."
  cd "$repo_dir"
  git pull
  cd .. 
else
  # Repository does not exist, clone it
  echo "Repository '$repo_dir' does not exist, cloning..."
  git clone "$repo_url" "$repo_dir"
fi

cd "$repo_dir"
pip install -e .[dev,testing]
pre-commit install
cd ..
#################### pyannote-audio #########################

#################### serab-byols #########################
repo_url="https://github.com/GasserElbanna/serab-byols.git"
repo_dir="serab-byols"

if [ -d "$repo_dir" ]; then
  # Repository exists, pull updates
  echo "Repository '$repo_dir' exists, pulling updates..."
  cd "$repo_dir"
  git pull
  cd .. 
else
  # Repository does not exist, clone it
  echo "Repository '$repo_dir' does not exist, cloning..."
  git clone "$repo_url" "$repo_dir"
fi

cd "$repo_dir"
pip install -e .
cd ..
#################### serab-byols #########################

#################### s3prl #########################
repo_url="https://github.com/s3prl/s3prl.git"
repo_dir="s3prl"

if [ -d "$repo_dir" ]; then
  # Repository exists, pull updates
  echo "Repository '$repo_dir' exists, pulling updates..."
  cd "$repo_dir"
  git pull
  cd .. 
else
  # Repository does not exist, clone it
  echo "Repository '$repo_dir' does not exist, cloning..."
  git clone "$repo_url" "$repo_dir"
fi

cd "$repo_dir"
pip install -e ".[all]"
cd ..
#################### s3prl #########################

#################### pycochleagram #########################
repo_url="https://github.com/mcdermottLab/pycochleagram.git"
repo_dir="pycochleagram"

if [ -d "$repo_dir" ]; then
  # Repository exists, pull updates
  echo "Repository '$repo_dir' exists, pulling updates..."
  cd "$repo_dir"
  git pull
  cd .. 
else
  # Repository does not exist, clone it
  echo "Repository '$repo_dir' does not exist, cloning..."
  git clone "$repo_url" "$repo_dir"
fi

cd "$repo_dir"
python setup.py install
cd ..
#################### pycochleagram #########################

#################### fab #########################
pip install -r requirements.txt
cd ../..
#################### fab #########################

echo "$env_name is ready to go!"

######################################## end #############################################


