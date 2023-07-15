import os
import inspect
from transformers import AutoProcessor, AutoModelForAudioClassification, AutoFeatureExtractor
import torch


class AudioEncoder:
    def __init__(self, model_name, model_checkpoint=None, models_save_dir=None, extra_params=None):
        """
        Initialize the AudioEncoder class.

        Args:
            model_checkpoint (str): The checkpoint name or path of the pretrained encoder model.
            models_save_dir (str): The directory where the pretrained models are saved.
        """
        # Check if temporal audio representation extraction method is available
        self.time_dependent_representation_available = self.check_temporal_audio_extraction()
        # Check if time independent audio representation extraction method is available
        self.time_independent_representation_available = self.check_time_independent_audio_extraction()

        if models_save_dir is None:
            models_save_dir = "../pretrained_models/"

        # Get the absolute path of the directory where the script is located
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Get the absolute path of the models save directory
        models_save_dir_absolute_path = os.path.join(script_directory, models_save_dir)

        # Load the pretrained encoder model
        self.encoder = AutoModelForAudioClassification.from_pretrained(model_checkpoint,
                                                                       cache_dir=models_save_dir_absolute_path)

        if bool(extra_params) and 'layer_number' in extra_params:
            self.layer_number = extra_params['layer_number']
        else:
            self.layer_number: None
        self.model_name = model_name

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
        # Encode the input waveforms to obtain embeddings
        if not hasattr(self, 'layer_number'):
            raise NotImplementedError("layer_number not specified.")
        embeddings = self.encoder(input_waveforms, output_hidden_states=True)
        if self.layer_number < 0 or self.layer_number > len(embeddings.hidden_states) - 1:
            raise NotImplementedError(
                f"layer_number can only be a value between 0 and {len(embeddings.hidden_states) - 1}")
        embeddings = embeddings.hidden_states[self.layer_number].permute(0, 2, 1)
        return embeddings
