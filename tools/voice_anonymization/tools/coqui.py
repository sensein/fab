# Import required libraries
from TTS.api import TTS  # TTS stands for Text-to-Speech but in this context, it's used for voice conversion
import torch
from tqdm import tqdm  # tqdm is used to create a progress bar for the conversion process
import torchaudio

class VoiceAnonymizer:
    def __init__(self, extra_params=None):
        """
        Initialize the VoiceAnonymizer.

        Parameters:
        - extra_params (dict, optional): A dictionary containing additional parameters for customization.
            - 'model_name' (str, optional): The name or path of the voice conversion model to use.
                                            Defaults to "voice_conversion_models/multilingual/vctk/freevc24".
            - 'progress_bar' (bool, optional): Whether to show a progress bar during conversion.
                                               Defaults to False.
        """
        # Check if the 'model_name' parameter is provided in the extra_params dictionary, otherwise use default
        if bool(extra_params) and 'model_name' in extra_params:
            self.model_name = extra_params['model_name']
        else:
            self.model_name = "voice_conversion_models/multilingual/vctk/freevc24"   

        # Check if the 'progress_bar' parameter is provided in the extra_params dictionary, otherwise use default (False)
        if bool(extra_params) and 'progress_bar' in extra_params:
            self.progress_bar = extra_params['progress_bar']
        else:
            self.progress_bar = False
            
        # Initialize the voice conversion model with the chosen model name and GPU availability
        self.tts = TTS(model_name=self.model_name, progress_bar=self.progress_bar, gpu=torch.cuda.is_available())

    def anonymize(self, source_files, target_files, output_files):
        """
        Perform voice anonymization on a list of source-target audio pairs.

        Parameters:
        - source_files (list of str): List of file paths for the source audio.
        - target_files (list of str): List of file paths for the target audio.
        - output_files (list of str): List of file paths where the converted output will be saved.

        Note:
        The length of source_files, target_files, and output_files should be the same.

        Output:
        - None: The method saves the converted audio files to the specified output paths.
        """
        # Loop through each triplet of source, target, and output file paths and convert the voices
        for line in tqdm(zip(source_files, target_files, output_files)):
            source_file, target_file, output_file = line

            # Perform voice conversion using the initialized TTS model and save the converted audio to the output file
            self.tts.voice_conversion_to_file(source_wav=source_file, target_wav=target_file, file_path=output_file)