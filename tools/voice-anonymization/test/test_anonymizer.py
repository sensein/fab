# TODO: in case you want to play with this code, please update source_files* and target_files* lists

import sys
import os

path_folder = '../'
script_directory = os.path.dirname(os.path.abspath(__file__))
path_folder_absolute_path = os.path.join(script_directory, path_folder)
# adding freeVC_folder to the system path
sys.path.insert(0, path_folder_absolute_path)

# Import the VoiceAnonymizer class from the anonymizer module
from anonymizer import VoiceAnonymizer

# Define the source, target, and output files for the first anonymization
source_files0 = ['./source.wav']
target_files0 = ['./target.wav']
output_files0 = ['./output0.wav']

# Define the source, target, and output files for the first anonymization
source_files1 = ['./source.wav']
target_files1 = ['./target.wav']
output_files1 = ['./output1.wav']

# Define the source, target, and output files for the second anonymization
source_files2 = ['./source.wav']
target_files2 = ['./target.wav']
output_files2 = ['./output2.wav']

# Define the source, target, and output files for the third anonymization
source_files3 = ['./source.wav']
output_files3 = ['./output3.wav']

# Define the source, target, and output files for the third anonymization
source_files4 = ['./source.wav']
output_files4 = ['./output4.wav']


# Create an instance of VoiceAnonymizer
anonymizer = VoiceAnonymizer()

# Anonymize using the speechT5 method
anonymizer.anonymize(method='speechT5', source_files=source_files0, target_files=target_files0, output_files=output_files0)

# Anonymize using the Coqui method
anonymizer.anonymize(method='coqui', source_files=source_files1, target_files=target_files1, output_files=output_files1)

# Anonymize using the FreeVC method
anonymizer.anonymize(method='freevc', source_files=source_files2, target_files=target_files2, output_files=output_files2)

# Anonymize using the McAdams method
anonymizer.anonymize(method='mcadams', source_files=source_files3, output_files=output_files3)

# TODO: test abdul-kareem's method