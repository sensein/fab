#!/usr/bin/env python
# coding: utf-8

from transformers import WhisperProcessor, WhisperForConditionalGeneration
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
            self.model_checkpoint = 'openai/whisper-large-v2'
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
        """
        if bool(extra_params) and 'word_timestamps' in extra_params:
            self.word_timestamps = extra_params['word_timestamps']
        else:
            self.word_timestamps = False
        """

        # Set the sampling rate for audio if provided, else set to the default (16000 Hz)
        if bool(extra_params) and 'sampling_rate' in extra_params:
            self.sampling_rate = extra_params['sampling_rate']
        else:
            self.sampling_rate = 16000

        # Determine the device (GPU or CPU) for computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.processor = WhisperProcessor.from_pretrained(self.model_checkpoint, cache_dir=self.models_save_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_checkpoint, cache_dir=self.models_save_dir)
        self.model.to(self.device)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")

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
            with torch.no_grad():
                input_features = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt", padding_strategy = 'max_length').input_features
                input_features = input_features.to(self.device) 
                predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)
                result = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                text = result[0]
                results.append(text)
                transcripts.append(text)
        return results, transcripts
