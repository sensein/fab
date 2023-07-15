import inspect
import serab_byols
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
            self.model_checkpoint = 'cvt'
            self.checkpoint_path = serab_byols.__file__.replace('serab_byols/__init__.py', '') + "checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth"
            self.cfg_path = serab_byols.__file__.replace('serab_byols/__init__.py', 'serab_byols/config.yaml')
        else:
            self.model_checkpoint = model_checkpoint
            self.checkpoint_path = extra_params['checkpoint_path'] or None
            self.cfg_path = extra_params['cfg_path'] or None

        self.encoder = serab_byols.load_model(self.checkpoint_path, self.model_checkpoint, self.cfg_path)

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

        embeddings = serab_byols.get_scene_embeddings(input_waveforms, self.encoder, self.cfg_path)
        return embeddings
