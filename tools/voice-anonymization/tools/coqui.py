# Import required libraries
from TTS.api import TTS  # TTS stands for Text-to-Speech but in this context, it's used for voice conversion
import torch
from tqdm import tqdm  # tqdm is used to create a progress bar for the conversion process


def anonymize(source_files, target_files, output_files,
              model_name="voice_conversion_models/multilingual/vctk/freevc24"):
    """
    Perform voice conversion on the provided source and target audio files and save the converted audio to output files.

    Parameters:
        source_files (list): List of paths to the source audio files (to be converted).
        target_files (list): List of paths to the target audio files (the desired speaker's voice).
        output_files (list): List of paths to save the converted output audio files.
        model_name (str): The name or path of the pre-trained voice conversion model.

    Returns:
        None: The function saves the converted audio files specified in the `output_files` list.
    """
    # Initialize the voice conversion model
    tts = TTS(model_name=model_name, progress_bar=False, gpu=torch.cuda.is_available())

    # Loop through each triplet of source, target, and output file paths and convert the voices
    for line in tqdm(zip(source_files, target_files, output_files)):
        source_file, target_file, output_file = line

        # Perform voice conversion and save the converted audio to the output file
        tts.voice_conversion_to_file(source_wav=source_file, target_wav=target_file, file_path=output_file)
