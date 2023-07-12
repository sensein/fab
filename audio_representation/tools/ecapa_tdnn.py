# TODO: AttributeError: 'EncoderClassifier' object has no attribute 'encode_temp_batch'
# TODO: comment the code!!!
# TODO: SETUP PRETRAINED MODELS FOLDER PATH!!!!


import inspect
from speechbrain.pretrained import EncoderClassifier

class AudioEncoder:
    def __init__(self, model_checkpoint="speechbrain/spkrec-ecapa-voxceleb"):
        self.time_dependent_representation_available = self.check_temporal_audio_extraction()
        self.time_independent_representation_available = self.check_time_independent_audio_extraction()
        self.classifier = EncoderClassifier.from_hparams(source=model_checkpoint)

    def check_temporal_audio_extraction(self):
        return 'temporal_audio_representation_extraction' in dir(self) and \
            inspect.ismethod(getattr(self, 'temporal_audio_representation_extraction'))

    def check_time_independent_audio_extraction(self):
        return 'time_independent_audio_representation_extraction' in dir(self) and \
            inspect.ismethod(getattr(self, 'time_independent_audio_representation_extraction'))

    def temporal_audio_representation_extraction(self, input_waveforms):
        if not self.time_dependent_representation_available:
            raise NotImplementedError("Temporal audio representation extraction is not available.")

        embeddings = self.classifier.encode_temp_batch(input_waveforms)
        return embeddings

    def time_independent_audio_representation_extraction(self, input_waveforms):
        if not self.time_independent_representation_available:
            raise NotImplementedError("Time independent audio representation extraction is not available.")

        embeddings = self.classifier.encode_batch(input_waveforms)
        return embeddings