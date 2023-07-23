#!/usr/bin/env python
# coding: utf-8

import whisper
import numpy as np
import torch
import pathlib
import os

class Transcriber:
    def __init__(self, model_name='whisper', model_checkpoint=None, language=None, models_save_dir=None, extra_params=None):
        """
        Initialize the Transcriber class.

        Args:
            model_name (str): The name of the Whisper ASR model. Default is 'whisper'.
            model_checkpoint (str): The checkpoint name for the Whisper model. Default is None, which uses 'large-v2'.
            language (str): The target language for transcription. Default is None.
            models_save_dir (str): The directory to save pretrained models. Default is None, which creates 'pretrained_models/whisper/' in the script's parent directory.
            extra_params (dict): Extra parameters for configuration. Default is None.

        Returns:
            None
        """
        self.model_name = model_name

        # Set the default model checkpoint if not provided
        if not model_checkpoint:
            self.model_checkpoint = 'large-v2'
        else:
            self.model_checkpoint = model_checkpoint

        # Set the target language for transcription if provided
        self.language = language

        # Set the directory to save pretrained models if not provided
        if models_save_dir is None:
            # Get the absolute path of the directory where the script is located
            script_directory = os.path.dirname(os.path.abspath(__file__))
            self.models_save_dir = os.path.join(script_directory, "../pretrained_models/whisper/") 
            pathlib.Path(self.models_save_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.models_save_dir = models_save_dir

        # Store any additional parameters provided
        self.extra_params = extra_params

        # Set word_timestamps to True if provided, else set to False (default)
        if bool(extra_params) and 'word_timestamps' in extra_params:
            self.word_timestamps = extra_params['word_timestamps']
        else:
            self.word_timestamps = False

        # Determine the device (GPU or CPU) for computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the Whisper ASR model
        self.model = whisper.load_model(self.model_checkpoint, device=self.device, download_root=self.models_save_dir)

        # Print information about the model's language and parameter count
        print(
            f"Model is {'multilingual' if self.model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in self.model.parameters()):,} parameters."
        )

    def transcribe(self, waveforms_or_files):
        """
        Transcribe the input audio waveforms or files.

        Args:
            waveforms_or_files (list): A list of audio waveforms or file paths.

        Returns:
            tuple: A tuple containing two lists:
                - results (list): A list of dictionaries containing the transcriptions and metadata for each input waveform or file.
                - transcripts (list): A list of strings containing transcriptions for each input waveform or file.
        """
        results = []
        transcripts = []
        for audio in waveforms_or_files:
            # Transcribe the audio using the Whisper model
            result = self.model.transcribe(audio, language=self.language, word_timestamps=self.word_timestamps)
            text = result['text']
            results.append(result)
            transcripts.append(text)
        return results, transcripts
