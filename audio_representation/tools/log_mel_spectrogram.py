import inspect
import torch
import torchaudio.transforms as T

class AudioEncoder:
    def __init__(self, model_name='log_mel_spectrogram', extra_params=None):
        """
        Initialize the AudioEncoder class.
        """
        # Check if temporal audio representation extraction method is available
        self.time_dependent_representation_available = self.check_temporal_audio_extraction()
        # Check if time independent audio representation extraction method is available
        self.time_independent_representation_available = self.check_time_independent_audio_extraction()

        if bool(extra_params) and 'sample_rate' in extra_params:
            self.sample_rate = extra_params['sample_rate']
        else:
            self.sample_rate = 16000

        if bool(extra_params) and 'n_fft' in extra_params:
            self.n_fft = extra_params['n_fft']
        else:
            self.n_fft = 4096

        if bool(extra_params) and 'win_length' in extra_params:
            self.win_length = extra_params['win_length']
        else:
            self.win_length = None

        if bool(extra_params) and 'hop_length' in extra_params:
            self.hop_length = extra_params['hop_length']
        else:
            self.hop_length = 512

        if bool(extra_params) and 'n_mels' in extra_params:
            self.n_mels = extra_params['n_mels']
        else:
            self.n_mels = 128

        if bool(extra_params) and 'f_min' in extra_params:
            self.f_min = extra_params['f_min']
        else:
            self.f_min = 5

        if bool(extra_params) and 'f_max' in extra_params:
            self.f_max = extra_params['f_max']
        else:
            self.f_max = 20000

        if bool(extra_params) and 'power' in extra_params:
            self.power = extra_params['power']
        else:
            self.power = 2

        # Load the pretrained encoder model
        self.encoder = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=self.power,
        )

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
        Extract temporal audio representations from input waveforms.

        Args:
            input_waveforms (Tensor): Input waveforms for which representations need to be extracted.

        Returns:
            Tensor: Temporal audio representations (embeddings) of the input waveforms.
        Raises:
            NotImplementedError: If temporal audio representation extraction is not available.
        """
        # Check if temporal audio representation extraction is available
        if not self.time_dependent_representation_available:
            raise NotImplementedError("Temporal audio representation extraction is not available.")
        mel_spectrogram = self.encoder(input_waveforms)
        embeddings = (mel_spectrogram + torch.finfo().eps).log().detach()
        return embeddings
