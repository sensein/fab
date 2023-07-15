import inspect
from pyannote.audio import Model, Inference
import torch


class AudioEncoder:
    def __init__(self, model_name='pyannote_audio', model_checkpoint=None, models_save_dir='', extra_params=None):
        """
        Initialize the AudioEncoder class.

        Args:
            model_name (str): Name of the audio model (default: 'pyannote_audio').
            model_checkpoint (str): Path or name of the model checkpoint file (default: None).
            models_save_dir (str): Directory to save the pretrained models (default: '').
            extra_params (dict): Extra parameters for the model (default: None).
        """

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
            if self.window == 'sliding':
                if bool(extra_params) and 'duration' in extra_params:
                    self.duration = extra_params['duration']
                else:
                    self.duration = 3.0
                if bool(extra_params) and 'step' in extra_params:
                    self.step = extra_params['step']
                else:
                    self.step = 1.0
        else:
            self.window = 'whole'

        self.encoder = Model.from_pretrained(model_checkpoint, use_auth_token=self.hf_token)
        if self.window == 'whole':
            self.inference = Inference(self.encoder, window=self.window)
        else:
            self.inference = Inference(self.encoder, window=self.window, duration=self.duration, step=self.step)

        # Check if temporal audio representation extraction method is available
        self.time_dependent_representation_available = self.check_temporal_audio_extraction()
        # Check if time independent audio representation extraction method is available
        self.time_independent_representation_available = self.check_time_independent_audio_extraction()

    def check_temporal_audio_extraction(self):
        """
        Check if temporal audio representation extraction method is available.

        Returns:
            bool: True if the method is available, False otherwise.
        """
        # Check if the method 'temporal_audio_representation_extraction' exists and is a method
        return 'temporal_audio_representation_extraction' in dir(self) and \
            inspect.ismethod(getattr(self, 'temporal_audio_representation_extraction')) and \
            self.window == 'sliding'

    def check_time_independent_audio_extraction(self):
        """
        Check if time independent audio representation extraction method is available.

        Returns:
            bool: True if the method is available, False otherwise.
        """
        # Check if the method 'time_independent_audio_representation_extraction' exists and is a method
        return 'time_independent_audio_representation_extraction' in dir(self) and \
            inspect.ismethod(getattr(self, 'time_independent_audio_representation_extraction')) and \
            self.window == 'whole'

    def time_independent_audio_representation_extraction(self, input_waveforms):
        """
        Extracts time-independent audio representations from input waveforms.

        Args:
            input_waveforms (torch.Tensor): Tensor of input waveforms of shape (batch_size, num_samples).

        Returns:
            embeddings_list (torch.Tensor): Tensor of extracted audio embeddings of shape (batch_size, num_embeddings, embedding_dim).
            embeddings_list (torch.Tensor): Tensor of extracted audio embeddings with shape (batch_size, embedding_dim, num_embeddings).
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
        return embeddings_list, embeddings_list

    def temporal_audio_representation_extraction(self, input_waveforms):
        """
        Extracts temporal audio representations from input waveforms.

        Args:
            input_waveforms (torch.Tensor): Tensor of input waveforms of shape (batch_size, num_samples).

        Returns:
            embeddings_list (torch.Tensor): Tensor of extracted audio embeddings of shape (batch_size, num_embeddings, embedding_dim).
            embeddings_list (torch.Tensor): Tensor of extracted audio embeddings with shape (batch_size, embedding_dim, num_embeddings).
        """
        if not self.time_dependent_representation_available:
            raise NotImplementedError("Temporal audio representation extraction is not available.")

        input_waveforms = input_waveforms.unsqueeze(1)
        embeddings_list = []
        for input_waveform in input_waveforms:
            embeddings_by_window = []
            embeddings = self.inference({"waveform": input_waveform, "sample_rate": self.sample_rate})
            for emb in embeddings:
                embeddings_by_window.append(emb[1])
            embeddings_list.append(torch.tensor(embeddings_by_window))
        embeddings_list = torch.stack(embeddings_list)
        return embeddings_list, embeddings_list.permute(0, 2, 1)


