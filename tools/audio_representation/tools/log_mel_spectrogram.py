import inspect
import torch
import torchaudio.transforms as T


class AudioEncoder:
    def __init__(self, model_name='log_mel_spectrogram', extra_params=None):
        """
        Initialize an AudioEncoder object.

        Args:
            model_name (str): The name of the audio encoding model. Default is 'log_mel_spectrogram'.
            extra_params (dict): Additional parameters for configuring the encoder. Default is None.

        Attributes:
            time_dependent_representation_available (bool): Flag indicating if temporal audio representation extraction method is available.
            time_independent_representation_available (bool): Flag indicating if time independent audio representation extraction method is available.
            sample_rate (int): The sample rate of the audio. Default is 16000.
            n_fft (int): The number of FFT points. Default is 4096.
            win_length (int or None): The window length in samples. Default is None.
            hop_length (int): The hop length in samples. Default is 512.
            n_mels (int): The number of Mel filterbanks. Default is 128.
            f_min (int): The minimum frequency in Hz. Default is 5.
            f_max (int): The maximum frequency in Hz. Default is 20000.
            power (int): The power of the spectrogram. Default is 2.
            encoder (torchaudio.transforms.MelSpectrogram): The pretrained encoder model.
        """
        self.time_dependent_representation_available = self.check_temporal_audio_extraction()
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
        return 'temporal_audio_representation_extraction' in dir(self) and \
            inspect.ismethod(getattr(self, 'temporal_audio_representation_extraction'))

    def check_time_independent_audio_extraction(self):
        """
        Check if time independent audio representation extraction method is available.

        Returns:
            bool: True if the method is available, False otherwise.
        """
        return 'time_independent_audio_representation_extraction' in dir(self) and \
            inspect.ismethod(getattr(self, 'time_independent_audio_representation_extraction'))

    def temporal_audio_representation_extraction(self, input_waveforms):
        """
        Extracts temporal audio representations from input waveforms.

        Args:
            input_waveforms (torch.Tensor): Input waveforms of shape (batch_size, num_samples).

        Returns:
            tuple: A tuple containing:
                - mel_spectrogram (torch.Tensor): Mel spectrogram of shape (batch_size, num_mel_bins, num_frames).
                - embeddings (torch.Tensor): Embeddings derived from the mel spectrogram of shape (batch_size, num_mel_bins, num_frames).
        """

        if not self.time_dependent_representation_available:
            raise NotImplementedError("Temporal audio representation extraction is not available.")

        # Extract mel spectrogram
        mel_spectrogram = self.encoder(input_waveforms)

        # Add epsilon and take the logarithm of the mel spectrogram
        embeddings = (mel_spectrogram + torch.finfo().eps).log().detach()

        # Return the mel spectrogram and embeddings
        return mel_spectrogram, embeddings

