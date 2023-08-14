import inspect
import serab_byols
import torch

class AudioEncoder:
    def __init__(self, model_name='pyannote_audio', model_checkpoint=None, models_save_dir='', extra_params=None):
        """
        Initialize the AudioEncoder class.

        Args:
            model_name (str): Name of the audio model.
            model_checkpoint (str): Path to the model checkpoint.
            models_save_dir (str): Directory to save the pretrained models.
            extra_params (dict): Extra parameters for model initialization.
        """
        self.time_dependent_representation_available = self.check_temporal_audio_extraction()
        self.time_independent_representation_available = self.check_time_independent_audio_extraction()

        if models_save_dir is None:
            models_save_dir = "../pretrained_models/"
        if model_checkpoint is None:
            self.model_checkpoint = 'cvt'
            self.checkpoint_path = serab_byols.__file__.replace('serab_byols/__init__.py',
                                                                '') + "checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth"
            self.cfg_path = serab_byols.__file__.replace('serab_byols/__init__.py', 'serab_byols/config.yaml')
        else:
            self.model_checkpoint = model_checkpoint
            self.checkpoint_path = extra_params['checkpoint_path'] or None
            self.cfg_path = extra_params['cfg_path'] or None

        if bool(extra_params) and 'frame_duration' in extra_params:
            self.frame_duration = extra_params['frame_duration']
        else:
            self.frame_duration = 1000
        if bool(extra_params) and 'hop_size' in extra_params:
            self.hop_size = extra_params['hop_size']
        else:
            self.hop_size = 1000
        self.encoder = serab_byols.load_model(self.checkpoint_path, self.model_checkpoint, self.cfg_path)

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

    def time_independent_audio_representation_extraction(self, input_waveforms):
        """
        Extracts time-independent audio representations from input waveforms.

        Args:
            input_waveforms (list): A list of input waveforms.

        Returns:
            tuple: A tuple containing the extracted embeddings.
                - embeddings (list): A list of audio embeddings.
        """

        # Check if time-independent representation is available
        if not self.time_independent_representation_available:
            raise NotImplementedError("Time independent audio representation extraction is not available.")

        with torch.no_grad():
            # Extract embeddings using the provided encoder and config path
            embeddings = serab_byols.get_scene_embeddings(input_waveforms, self.encoder, self.cfg_path)

        # Return the extracted embeddings
        return embeddings, embeddings

    # TODO: it looks like serab_byols.get_timestamp_embeddings doesn't work properly
    # It says: "An error has occurred :  "
    '''
    def temporal_audio_representation_extraction(self, input_waveforms):
        """
        Extract temporal audio representations from input waveforms.

        Args:
            input_waveforms (torch.Tensor): Input waveforms of shape (batch_size, channels, samples).

        Returns:
            torch.Tensor: Extracted audio representations of shape (batch_size, features).

        Raises:
            NotImplementedError: If temporal audio representation extraction is not available.
        """
        # Check if temporal audio representation extraction is available
        if not self.time_dependent_representation_available:
            raise NotImplementedError("Temporal audio representation extraction is not available.")

        embeddings, timestamps = serab_byols.get_timestamp_embeddings(input_waveforms, self.encoder, self.frame_duration, self.hop_size)
        return embeddings
    '''
