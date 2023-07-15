import torch
from tools.ecapa_tdnn import AudioEncoder as SpkrecEcapaVoxceleb
from tools.wavLM_HuBERT_wav2vec2_data2vec import AudioEncoder as WavLMHuBERTWav2vec2Data2vec
from tools.human_cochleagram import AudioEncoder as HumanCochleagram
from tools.log_mel_spectrogram import AudioEncoder as LogMelSpectrogram
from tools.s3prl_apc_tera import AudioEncoder as S3prl
from tools.whisper import AudioEncoder as Whisper
from tools.pyannote_audio_speaker_embeddings import AudioEncoder as PyannoteAudioSpeakerEmbeddings
from tools.byol import AudioEncoder as Byol

'''
elif model_name == "Whisper":
    # If the model_name is "FacebookModel", instantiate the model accordingly
    self.model = Whisper(model_name=model_name, model_checkpoint=model_checkpoint, models_save_dir=models_save_dir,
                         extra_params=extra_params)
'''
'''
elif model_name == "PyannoteAudioSpeakerEmbeddings":
    # If the model_name is "FacebookModel", instantiate the model accordingly
    self.model = PyannoteAudioSpeakerEmbeddings(model_name=model_name, model_checkpoint=model_checkpoint,
                                                models_save_dir=models_save_dir,
                                                extra_params=extra_params)
'''


class AudioRepresentation:
    def __init__(self, model_name, model_checkpoint=None, models_save_dir=None, extra_params=None):
        """
        Initialize the AudioRepresentation class with the specified model_name and model_checkpoint.

        Args:
            model_name (str): Name of the audio encoding model.
            model_checkpoint (str, optional): Path to the model checkpoint file. Defaults to None.
        """
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint

        if model_name.startswith("SpkrecEcapaVoxceleb"):
            # If the model_name is "SpkrecEcapaVoxceleb", instantiate the model accordingly
            self.model = SpkrecEcapaVoxceleb(model_name=model_name, model_checkpoint=model_checkpoint,
                                             models_save_dir=models_save_dir,
                                             extra_params=extra_params)
        elif model_name.startswith("wavLM") or model_name.startswith("HuBERT") or model_name.startswith(
                "wav2vec2") or model_name.startswith("data2vec"):
            # If the model_name is "FacebookModel", instantiate the model accordingly
            self.model = WavLMHuBERTWav2vec2Data2vec(model_name=model_name, model_checkpoint=model_checkpoint,
                                                     models_save_dir=models_save_dir,
                                                     extra_params=extra_params)
        elif model_name.startswith("apc") or model_name.startswith("tera"):
            # If the model_name is "FacebookModel", instantiate the model accordingly
            self.model = S3prl(model_name=model_name, model_checkpoint=model_checkpoint,
                               models_save_dir=models_save_dir,
                               extra_params=extra_params)
        elif model_name.startswith('HumanCochleagram'):
            self.model = HumanCochleagram(model_name=model_name, extra_params=extra_params)
        elif model_name.startswith('LogMelSpectrogram'):
            self.model = LogMelSpectrogram(model_name=model_name, extra_params=extra_params)
        elif model_name.startswith("Whisper"):
            self.model = Whisper(model_name=model_name, model_checkpoint=model_checkpoint,
                                 models_save_dir=models_save_dir,
                                 extra_params=extra_params)
        elif model_name.startswith("pyannote_audio"):
            self.model = PyannoteAudioSpeakerEmbeddings(model_name=model_name, model_checkpoint=model_checkpoint,
                                                        models_save_dir=models_save_dir,
                                                        extra_params=extra_params)
        elif model_name.startswith("byol"):
            self.model = Byol(model_name=model_name, model_checkpoint=model_checkpoint,
                                                        models_save_dir=models_save_dir,
                                                        extra_params=extra_params)
        else:
            # Raise a ValueError if an invalid model is specified
            raise ValueError("Invalid model specified.")

        self.temporal_encoding_exists = self.model.check_temporal_audio_extraction()
        self.pooled_temporal_encoding_exists = self.model.check_temporal_audio_extraction()
        self.contextual_encoding_exists = self.model.check_time_independent_audio_extraction()

    def temporal_encoding(self, input_waveforms):
        """
        Perform temporal encoding on the input waveforms using the instantiated model.

        Args:
            input_waveforms (torch.Tensor): Input waveforms to encode.

        Returns:
            torch.Tensor: Encoded audio representations.
        """
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement temporal audio encoding.")
        # Perform temporal encoding on the input waveforms using the model
        return self.model.temporal_audio_representation_extraction(input_waveforms)

    def contextual_encoding(self, input_waveforms):
        """
        Perform contextual encoding on the input waveforms using the instantiated model.

        Args:
            input_waveforms (torch.Tensor): Input waveforms to encode.

        Returns:
            torch.Tensor: Encoded audio representations.
        """
        if not self.model.check_time_independent_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement contextual audio encoding.")
        # Perform contextual encoding on the input waveforms using the model
        return self.model.time_independent_audio_representation_extraction(input_waveforms)

    def pooled_temporal_encoding(self, input_waveforms):
        """
        Perform temporal encoding on the input waveforms using the instantiated model and apply various pooling options.

        Args:
            input_waveforms (torch.Tensor): Input waveforms to encode.

        Returns:
            dict: Dictionary containing pooled embeddings.
        """
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement temporal audio encoding.")

        # Perform temporal encoding on the input waveforms using the model
        embeddings = self.model.temporal_audio_representation_extraction(input_waveforms)

        # Perform all pooling options
        mean_pool = torch.mean(embeddings, dim=2)
        max_pool = torch.max(embeddings, dim=2)[0]
        min_pool = torch.min(embeddings, dim=2)[0]
        sum_pool = torch.sum(embeddings, dim=2)

        result = []
        for i in range(embeddings.shape[0]):
            result.append({
                'global_min_pooling': min_pool[i],
                'global_max_pooling': max_pool[i],
                'global_mean_pooling': mean_pool[i],
                'global_mean_plus_max_pooling': mean_pool[i] + max_pool[i],
                'global_sum_pooling': sum_pool[i],
            })
        return result

    def temporal_encoding_size(self):
        """
        Get the shape of the temporal encoding output.

        Returns:
            torch.Size: Shape of the temporal encoding output.
        """
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement temporal audio encoding.")
        return self.temporal_encoding(torch.rand(1, 1, 16000).squeeze(1)).shape[1]

    def contextual_encoding_size(self):
        """
        Get the size (length) of the contextual encoding output.

        Returns:
            int: Size of the contextual encoding output.
        """
        if not self.model.check_time_independent_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement contextual audio encoding.")
        return self.contextual_encoding(torch.rand(1, 1, 16000).squeeze(1)).shape[1]

    def pooled_temporal_encoding_size(self):
        """
        Get the size (length) of the contextual encoding output.

        Returns:
            int: Size of the contextual encoding output.
        """
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement temporal audio encoding.")
        embeddings_dict = self.pooled_temporal_encoding(torch.rand(1, 1, 16000).squeeze(1))[0]
        first_dict_key = next(iter(embeddings_dict))
        return embeddings_dict[first_dict_key].shape[0]
