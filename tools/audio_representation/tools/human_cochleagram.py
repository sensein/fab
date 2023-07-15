import inspect
import pycochleagram.cochleagram as cgram
import torch

class AudioEncoder:
    def __init__(self, model_name='human_cochleagram', extra_params=None):
        """
        Initialize the AudioEncoder.

        Args:
            model_name (str): Name of the encoder model to use. Default is 'human_cochleagram'.
            extra_params (dict): Extra parameters for the encoder. Optional.

        """
        # Check if temporal audio representation extraction method is available
        self.time_dependent_representation_available = self.check_temporal_audio_extraction()
        # Check if time independent audio representation extraction method is available
        self.time_independent_representation_available = self.check_time_independent_audio_extraction()

        # Load the pretrained encoder model
        self.encoder = cgram.human_cochleagram

        if bool(extra_params) and 'sample_rate' in extra_params:
            self.sample_rate = extra_params['sample_rate']
        else:
            self.sample_rate = 16000
        if bool(extra_params) and 'strict' in extra_params:
            self.strict = extra_params['strict']
        else:
            self.strict = False
        if bool(extra_params) and 'n' in extra_params:
            self.n = extra_params['n']
        else:
            self.n = 40

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

    def temporal_audio_representation_extraction(self, input_waveforms):
        """
        Extracts temporal audio representations from input waveforms.

        Args:
            input_waveforms (numpy.ndarray): Input audio waveforms.

        Returns:
            cochleagram (numpy.ndarray): Temporal audio representation in the form of a cochleagram.
            embeddings (torch.Tensor): Temporal audio representations as embeddings.

        Raises:
            NotImplementedError: If temporal audio representation extraction is not available.
        """

        # Check if temporal audio representation extraction is available
        if not self.time_dependent_representation_available:
            raise NotImplementedError("Temporal audio representation extraction is not available.")

        # Extract cochleagram using the encoder
        cochleagram = self.encoder(input_waveforms, sr=self.sample_rate, strict=self.strict, n=self.n)

        # Convert cochleagram to log scale and detach from computation graph
        embeddings = (torch.from_numpy(cochleagram) + torch.finfo().eps).log().cpu().detach()

        # If embeddings have a dimension of 2, add an additional dimension
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        return cochleagram, embeddings
