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

# Define the source, target, and output files for the 4th anonymization
source_files4 = ['./source.wav']
output_files4 = ['./output4.wav']


# Anonymize using the speechT5 method
anonymizer = VoiceAnonymizer(method='speechT5')
anonymizer.anonymize(source_files=source_files0, target_files=target_files0, output_files=output_files0)

# Anonymize using the Coqui method
anonymizer = VoiceAnonymizer(method='coqui')
anonymizer.anonymize(source_files=source_files1, target_files=target_files1, output_files=output_files1)

# Anonymize using the FreeVC method
anonymizer = VoiceAnonymizer(method='freevc')
anonymizer.anonymize(source_files=source_files2, target_files=target_files2, output_files=output_files2)

# Anonymize using the McAdams method
anonymizer = VoiceAnonymizer(method='mcadams')
anonymizer.anonymize(source_files=source_files3, output_files=output_files3)
'''
# Anonymize using the PIR method
#anonymizer = VoiceAnonymizer(method='pir')
#anonymizer.anonymize(source_files=source_files4, output_files=output_files4)
'''