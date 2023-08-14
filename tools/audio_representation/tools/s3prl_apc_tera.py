import os
import inspect
import s3prl.hub as s3hub
import torch


class AudioEncoder:
    def __init__(self, model_name, model_checkpoint=None, models_save_dir=None, extra_params=None):
        """
        Initialize the AudioEncoder object.

        Args:
            model_name (str): Name of the audio encoding model.
            model_checkpoint (str): Checkpoint name for the model. Default is None.
            models_save_dir (str): Directory where the pretrained models are saved. Default is None.
            extra_params (dict): Extra parameters for the audio encoder. Default is None.
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

        self.model_name = model_name
        self.extra_params = extra_params

        # Set the default model checkpoint based on the model name
        if model_checkpoint is None and self.model_name == 'apc':
            model_checkpoint = 'apc'
        elif model_checkpoint is None and self.model_name == 'tera':
            model_checkpoint = 'tera'

        if bool(extra_params) and 'layer_number' in extra_params:
            self.layer_number = extra_params['layer_number']
        else:
            # the last layer is the default one
            self.layer_number = -1

        # Instantiate the audio encoder model
        self.encoder = getattr(s3hub, model_checkpoint)()
        self.encoder.eval()

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
        Extracts temporal audio representation from input waveforms.

        Args:
            input_waveforms (Tensor): Input waveforms for audio processing.

        Returns:
            tuple: A tuple containing raw_encoder_response and embeddings.
                - raw_encoder_response (dict): Raw response from the encoder.
                - embeddings (Tensor): Extracted embeddings from the encoder.

        Raises:
            NotImplementedError: If temporal audio representation extraction is not available.
            NotImplementedError: If layer_number is invalid.
        """

        if not self.time_dependent_representation_available:
            raise NotImplementedError("Temporal audio representation extraction is not available.")

        with torch.no_grad():
            raw_encoder_response = self.encoder(input_waveforms)

        # Encode the input waveforms to obtain embeddings
        if self.layer_number > len(raw_encoder_response["hidden_states"]) - 1:
            raise NotImplementedError(
                f"layer_number can only be a value between 0 and {len(raw_encoder_response['hidden_states']) - 1}")

        embeddings = raw_encoder_response["hidden_states"][self.layer_number].permute(0, 2, 1)

        return raw_encoder_response, embeddings

