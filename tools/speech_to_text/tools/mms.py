#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
import pathlib
import os

class Transcriber:
    def __init__(self, model_name='mms', model_checkpoint=None, language=None, models_save_dir=None, extra_params=None):
        """
        Initialize the Transcriber class.

        Args:
            model_name (str): The name of the Wav2Vec2 model. Default is 'mms'.
            model_checkpoint (str): The Hugging Face model checkpoint. Default is None, which uses 'facebook/mms-1b-all'.
            language (str): The target language for transcription. Default is None.
            models_save_dir (str): The directory to save pretrained models. Default is None, which creates 'pretrained_models/' in the script's parent directory.
            extra_params (dict): Extra parameters for configuration. Default is None.

        Returns:
            None
        """
        self.model_name = model_name

        # Set the default model checkpoint if not provided
        if not model_checkpoint:
            self.model_checkpoint = 'facebook/mms-1b-all'
        else:
            self.model_checkpoint = model_checkpoint

        # Set the target language for the tokenizer if provided
        self.language = language

        # Set the directory to save pretrained models if not provided
        if models_save_dir is None:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            self.models_save_dir = os.path.join(script_directory, "../pretrained_models/") 
            pathlib.Path(self.models_save_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.models_save_dir = models_save_dir

        # Store any additional parameters provided
        self.extra_params = extra_params

        # Determine the device (GPU or CPU) for computation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the processor and model from the Hugging Face's transformers library
        self.processor = AutoProcessor.from_pretrained(self.model_checkpoint, cache_dir=self.models_save_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_checkpoint, cache_dir=self.models_save_dir).to(self.device)

        # Set the sampling rate for audio if provided, else set to the default (16000 Hz)
        if bool(extra_params) and 'sampling_rate' in extra_params:
            self.sampling_rate = extra_params['sampling_rate']
        else:
            self.sampling_rate = 16000

        # Load the target language if provided
        if hasattr(self, 'language') and self.language is not None:
            self.processor.tokenizer.set_target_lang(self.language)
            self.model.load_adapter(self.language)

    def transcribe(self, waveforms):
        """
        Transcribe the input speech waveforms.

        Args:
            waveforms (list): A list of audio waveforms as input.

        Returns:
            tuple: A tuple containing two lists:
                - results (list): A list of tensors containing model outputs for each input waveform.
                - transcripts (list): A list of strings containing transcriptions for each input waveform.
        """
        results = []
        transcripts = []
        for audio in waveforms:
            inputs = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs).logits
            ids = torch.argmax(outputs, dim=-1)[0]
            transcription = self.processor.decode(ids)
            results.append(outputs)
            transcripts.append(transcription)
        return results, transcripts
