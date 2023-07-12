from tools.ecapa_tdnn import AudioEncoder as SpkrecEcapaVoxceleb


class AudioRepresentation:

    def __init__(self, model_name, model_checkpoint=None):
        # Initialize the AudioRepresentation class with the specified model_name and model_checkpoint
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint

        if model_name == "SpkrecEcapaVoxceleb":
            # If the model_name is "SpkrecEcapaVoxceleb", instantiate the model accordingly
            if model_checkpoint:
                # If a model_checkpoint is provided, create the model using the checkpoint
                self.model = SpkrecEcapaVoxceleb(model_checkpoint=model_checkpoint)
            else:
                # If no model_checkpoint is provided, create the model without any checkpoint
                self.model = SpkrecEcapaVoxceleb()
        else:
            # Raise a ValueError if an invalid model is specified
            raise ValueError("Invalid model specified.")

    def temporal_encoding(self, input_waveforms):
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {model_name} does not implent temporal audio encoding.")
        # Perform temporal encoding on the input waveforms using the model
        return self.model.temporal_audio_representation_extraction(input_waveforms)

    def contextual_encoding(self, input_waveforms):
        if not self.model.check_time_independent_audio_extraction():
            raise ValueError(f"Model {model_name} does not implent contextual audio encoding.")
        # Perform contextual encoding on the input waveforms using the model
        return self.model.time_independent_audio_representation_extraction(input_waveforms)