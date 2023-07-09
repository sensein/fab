source_files = ['./target.wav']
target_files = ['./target.wav']
output_files = ['./output.wav']

#from tools.coqui import anonymize
#anonymize(source_files=source_files, target_files=target_files, output_files=output_files)

#from tools.freevc import anonymize
#anonymize(source_files=source_files, target_files=target_files, output_files=output_files)

from tools.mcadams import anonymize
anonymize(source_files=source_files, output_files=output_files)
