#!/bin/bash

#################### config #########################
# Source the config file
source config.sh
tools_folder="tools"
#################### config #########################

######################################## start #############################################

echo "Hello world!"

if conda info --envs | grep -q "/$env_name$"; then
  echo "Environment '$env_name' already exists"
else
  echo "Environment '$env_name' does not exist"
  conda create -n $env_name python=3.9 -y
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

#################### speechbrain #########################
repo_url="https://github.com/fabiocat93/speechbrain.git"
repo_dir="speechbrain"

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
pip install -r requirements.txt
pip install --editable .
cd ..
#################### speechbrain #########################

#################### s3prl #########################
repo_url="https://github.com/fabiocat93/s3prl.git"
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


#################### pycochleagram #########################
repo_url="https://github.com/fabiocat93/pycochleagram.git"
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

#################### TTS #########################
repo_url="https://github.com/coqui-ai/TTS"
repo_dir="TTS"

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
#################### TTS #########################

#################### FreeVC #########################
repo_url="https://github.com/fabiocat93/FreeVC.git"
repo_dir="FreeVC"

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
pip install -r requirements.txt

url1="https://www.dropbox.com/scl/fi/tqz5z98nxn0voba223z6a/logs.zip?rlkey=uivc6sbd8lie0iw2bds1wfrwa&dl=1"  # URL of the directory to download
url2="https://www.dropbox.com/scl/fi/sqsmafd8botn3fb1y2wy8/checkpoints.zip?rlkey=wq79puddc7x3vqibe3cenajgl&dl=1"  # URL of the directory to download
url3="https://www.dropbox.com/scl/fi/abe95mw01hr87h2fx3i76/WavLM-Large.pt.zip?rlkey=0po2jd2u77q2zt6irrmmz2shc&dl=1"  # URL of the file to download

destination_folder1="logs"         # Destination folder path
destination_folder2="checkpoints"         # Destination folder path
destination_folder3="wavlm"           # Destination folder path

# Check if the destination folder exists
if [ ! -d "$destination_folder1" ]; then
  wget -O $destination_folder1.zip "$url1"
  unzip $destination_folder1.zip
  rm $destination_folder1.zip
else
  echo "Folder $destination_folder1 already exists."
fi

# Check if the destination folder exists
if [ ! -d "$destination_folder2" ]; then
  wget -O $destination_folder2.zip "$url2"
  unzip $destination_folder2.zip
  rm $destination_folder2.zip
else
  echo "Folder $destination_folder2 already exists."
fi

file_name="WavLM-Large.pt"
file_path="$destination_folder3/$file_name"     # Full path of the destination file
if [ -f "$file_path" ]; then
  echo "File $file_name already exists in $folder_path."
else
  echo "File $file_name does not exist. Downloading..."
  wget -O $file_name.zip "$url3"
  unzip $file_name.zip
  rm $file_name.zip
  mv $file_name $file_path
fi

# shellcheck disable=SC2103
cd ..
#################### FreeVC #########################

#################### fab #########################
pip install -r requirements.txt
cd ../..
#################### fab #########################

echo "$env_name is ready to go!"
echo "To start working with it, please do 'conda activate $env_name'"
######################################## end #############################################


