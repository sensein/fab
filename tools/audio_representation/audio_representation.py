import torch
from .tools.ecapa_tdnn import AudioEncoder as EcapaTDNN
from .tools.wavLM_HuBERT_wav2vec2_data2vec import AudioEncoder as WavLMHuBERTWav2vec2Data2vec
from .tools.human_cochleagram import AudioEncoder as HumanCochleagram
from .tools.log_mel_spectrogram import AudioEncoder as LogMelSpectrogram
from .tools.s3prl_apc_tera import AudioEncoder as S3prl
from .tools.whisper import AudioEncoder as Whisper
#from .tools.pyannote_audio_speaker_embeddings import AudioEncoder as PyannoteAudioSpeakerEmbeddings
from .tools.byol import AudioEncoder as Byol

class AudioRepresentation:
    def __init__(self, model_name, model_checkpoint=None, models_save_dir=None, extra_params=None):
        """
        Initialize the AudioRepresentation class.

        Args:
            model_name (str): Name of the audio model.
            model_checkpoint (str): Path to the model checkpoint file (default: None).
            models_save_dir (str): Directory to save the models (default: None).
            extra_params (dict): Extra parameters for model initialization (default: None).
        """
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint
        
        # Instantiate the appropriate audio encoder based on the model_name
        if model_name.startswith("EcapaTDNN"):
            self.model = EcapaTDNN(model_name=model_name, model_checkpoint=model_checkpoint,
                                   models_save_dir=models_save_dir, extra_params=extra_params)
        elif model_name.startswith("wavLM") or model_name.startswith("HuBERT") or model_name.startswith(
                "wav2vec2") or model_name.startswith("data2vec"):
            self.model = WavLMHuBERTWav2vec2Data2vec(model_name=model_name, model_checkpoint=model_checkpoint,
                                                     models_save_dir=models_save_dir, extra_params=extra_params)
        elif model_name.startswith("apc") or model_name.startswith("tera"):
            self.model = S3prl(model_name=model_name, model_checkpoint=model_checkpoint,
                               models_save_dir=models_save_dir, extra_params=extra_params)
        elif model_name.startswith('HumanCochleagram'):
            self.model = HumanCochleagram(model_name=model_name, extra_params=extra_params)
        elif model_name.startswith('LogMelSpectrogram'):
            self.model = LogMelSpectrogram(model_name=model_name, extra_params=extra_params)
        elif model_name.startswith("Whisper"):
            self.model = Whisper(model_name=model_name, model_checkpoint=model_checkpoint,
                                 models_save_dir=models_save_dir, extra_params=extra_params)
        elif model_name.startswith("Byol"):
            self.model = Byol(model_name=model_name, model_checkpoint=model_checkpoint,
                              models_save_dir=models_save_dir, extra_params=extra_params)
        else:
            # Raise a ValueError if an invalid model is specified
            raise ValueError("Invalid model specified.")

        '''
        elif model_name.startswith("pyannote_audio"):
            self.model = PyannoteAudioSpeakerEmbeddings(model_name=model_name, model_checkpoint=model_checkpoint,
                                                    models_save_dir=models_save_dir, extra_params=extra_params)

        '''

        # Check the availability of different encoding options
        self.temporal_encoding_exists = self.model.check_temporal_audio_extraction()
        self.pooled_temporal_encoding_exists = self.model.check_temporal_audio_extraction()
        self.contextual_encoding_exists = self.model.check_time_independent_audio_extraction()

    def temporal_encoding(self, input_waveforms):
        """
        Perform temporal encoding on the input waveforms using the model.

        Args:
            input_waveforms (torch.Tensor): Input waveforms for encoding. Shape: (batch_size, num_channels, num_samples).

        Returns:
            torch.Tensor: Temporal audio representations. Shape: (batch_size, encoding_size).
        """
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement temporal audio encoding.")
        return self.model.temporal_audio_representation_extraction(input_waveforms)

    def contextual_encoding(self, input_waveforms):
        """
        Perform contextual encoding on the input waveforms using the model.

        Args:
            input_waveforms (torch.Tensor): Input waveforms for encoding. Shape: (batch_size, num_channels, num_samples).

        Returns:
            torch.Tensor: Contextual audio representations. Shape: (batch_size, encoding_size).
        """
        if not self.model.check_time_independent_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement contextual audio encoding.")
        return self.model.time_independent_audio_representation_extraction(input_waveforms)

    def pooled_temporal_encoding(self, input_waveforms):
        """
        Perform pooled temporal encoding on the input waveforms using the model.

        Args:
            input_waveforms (torch.Tensor): Input waveforms for encoding. Shape: (batch_size, num_channels, num_samples).

        Returns:
            list: List of dictionaries containing different pooling options.
                Each dictionary contains the following keys:
                - 'global_min_pooling': Min pooling result. Shape: (encoding_size,).
                - 'global_max_pooling': Max pooling result. Shape: (encoding_size,).
                - 'global_mean_pooling': Mean pooling result. Shape: (encoding_size,).
                - 'global_mean_plus_max_pooling': Sum of mean and max pooling results. Shape: (encoding_size,).
                - 'global_sum_pooling': Sum pooling result. Shape: (encoding_size,).
        """
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement temporal audio encoding.")

        # Perform temporal encoding on the input waveforms using the model
        raw_encoder_response, embeddings = self.model.temporal_audio_representation_extraction(input_waveforms)

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

        return raw_encoder_response, result

    def temporal_encoding_size(self):
        """
        Get the size (length) of the temporal encoding output.

        Returns:
            torch.Size: Shape of the temporal encoding output.
        """
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement temporal audio encoding.")
        raw_encoder_response, embeddings = self.temporal_encoding(torch.rand(1, 1, 16000).squeeze(1))
        return embeddings.shape[1]

    def contextual_encoding_size(self):
        """
        Get the size (length) of the contextual encoding output.

        Returns:
            int: Size of the contextual encoding output.
        """
        if not self.model.check_time_independent_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement contextual audio encoding.")
        raw_encoder_response, embeddings = self.contextual_encoding(torch.rand(1, 1, 16000).squeeze(1))
        return embeddings.shape[1]

    def pooled_temporal_encoding_size(self):
        """
        Get the size (length) of the pooled temporal encoding output.

        Returns:
            int: Size of the pooled temporal encoding output.
        """
        if not self.model.check_temporal_audio_extraction():
            raise ValueError(f"Model {self.model_name} does not implement temporal audio encoding.")
        raw_encoder_response, embeddings_dict = self.pooled_temporal_encoding(torch.rand(1, 1, 16000).squeeze(1))
        first_dict_key = next(iter(embeddings_dict[0]))
        return embeddings_dict[0][first_dict_key].shape[0]