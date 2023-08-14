import os
import inspect
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch


class AudioEncoder:
    def __init__(self, model_name, model_checkpoint=None, models_save_dir=None, extra_params=None):
        """
        Initialize the AudioEncoder class.

        Args:
            model_name (str): Name of the audio encoder model.
            model_checkpoint (str): Path or identifier of the pretrained model checkpoint. Default is None.
            models_save_dir (str): Directory to save the pretrained models. Default is None.
            extra_params (dict): Additional parameters for the audio encoder. Default is None.
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
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint,
                                                                      cache_dir=models_save_dir_absolute_path)

        if bool(extra_params) and 'layer_number' in extra_params:
            self.layer_number = extra_params['layer_number']
        else:
            # the last layer is the default one
            self.layer_number = -1
        if bool(extra_params) and 'sampling_rate' in extra_params:
            self.sampling_rate = extra_params['sampling_rate']
        else:
            self.sampling_rate = 16000

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
        Extracts temporal audio representations from input waveforms.

        Args:
            input_waveforms (list): List of input audio waveforms.

        Returns:
            raw_encoder_response (list): List of raw encoder responses for each input waveform.
            output_embeddings (torch.Tensor): Tensor containing output embeddings for each input waveform.
        """

        # Check if temporal audio representation extraction is available
        if not self.time_dependent_representation_available:
            raise NotImplementedError("Temporal audio representation extraction is not available.")
        if not hasattr(self, 'layer_number'):
            raise NotImplementedError("layer_number not specified.")

        raw_encoder_response = []
        output_embeddings = []

        for input_waveform in input_waveforms:
            # Encode the input waveforms to obtain embeddings
            with torch.no_grad():
                inputs = self.feature_extractor(input_waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
                input_features = inputs.input_features
                embeddings = self.encoder(input_features, output_hidden_states=True)
            raw_encoder_response.append(embeddings)

            if self.layer_number > len(embeddings.hidden_states) - 1:
                raise NotImplementedError(
                    f"layer_number can only be a value between 0 and {len(embeddings.hidden_states) - 1}")

            embeddings = embeddings.hidden_states[self.layer_number].squeeze(0).permute(1, 0)
            output_embeddings.append(embeddings)

        output_embeddings = torch.stack(output_embeddings)

        return raw_encoder_response, output_embeddings