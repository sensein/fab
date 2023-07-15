import inspect
from pyannote.audio import Model, Inference

import torch


class AudioEncoder:
    def __init__(self, model_name='pyannote_audio', model_checkpoint=None, models_save_dir='', extra_params=None):
        """
        Initialize the AudioEncoder class.
        """
        # Check if temporal audio representation extraction method is available
        self.time_dependent_representation_available = self.check_temporal_audio_extraction()
        # Check if time independent audio representation extraction method is available
        self.time_independent_representation_available = self.check_time_independent_audio_extraction()

        if models_save_dir is None:
            models_save_dir = "../pretrained_models/"
        if model_checkpoint is None:
            model_checkpoint = "pyannote/embedding"

        # Load the pretrained encoder model
        if bool(extra_params) and 'sample_rate' in extra_params:
            self.sample_rate = extra_params['sample_rate']
        else:
            self.sample_rate = 16000
        if bool(extra_params) and 'hf_token' in extra_params:
            self.hf_token = extra_params['hf_token']
        else:
            self.hf_token = None
        if bool(extra_params) and 'window' in extra_params:
            self.window = extra_params['window']
        else:
            self.window = 'whole'

        self.encoder = Model.from_pretrained(model_checkpoint, use_auth_token=self.hf_token)
        self.inference = Inference(self.encoder, window=self.window)

    def check_temporal_audio_extraction(self):
        """
        Check if temporal audio representation extraction method is available.

        Returns:
            bool: True if the method is available, False otherwise.
        """
        # Check if the method 'temporal_audio_representation_extraction' exists and is a method
        return 'temporal_audio_representation_extraction' in dir(self) and \
            inspect.ismethod(getattr(self, 'temporal_audio_representation_extraction'))

    def check_time_independent_audio_extraction(self):
        """
        Check if time independent audio representation extraction method is available.

        Returns:
            bool: True if the method is available, False otherwise.
        """
        # Check if the method 'time_independent_audio_representation_extraction' exists and is a method
        return 'time_independent_audio_representation_extraction' in dir(self) and \
            inspect.ismethod(getattr(self, 'time_independent_audio_representation_extraction'))

    def time_independent_audio_representation_extraction(self, input_waveforms):
        """
        Extract time independent audio representations from input waveforms.

        Args:
            input_waveforms (Tensor): Input waveforms for which representations need to be extracted.

        Returns:
            Tensor: Time independent audio representations (embeddings) of the input waveforms.
        Raises:
            NotImplementedError: If time independent audio representation extraction is not available.
        """
        # Check if time independent audio representation extraction is available
        if not self.time_independent_representation_available:
            raise NotImplementedError("Time independent audio representation extraction is not available.")

        input_waveforms = input_waveforms.unsqueeze(1)
        embeddings_list = []
        for input_waveform in input_waveforms:
            embeddings = self.inference({"waveform": input_waveform, "sample_rate": self.sample_rate})
            embeddings_list.append(torch.tensor(embeddings))
        embeddings_list = torch.stack(embeddings_list)
        return embeddings_list
