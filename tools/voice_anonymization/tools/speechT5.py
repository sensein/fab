# Import necessary libraries
import sys
import os
import soundfile as sf
import torchaudio
from tqdm import tqdm
import torch

# Import modules from the audio_representation library
from tools.audio_representation import AudioRepresentation

# Import models from the transformers library
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan


class VoiceAnonymizer:
    def __init__(self, extra_params=None):        
        if bool(extra_params) and 'processor_name' in extra_params:
            self.processor_name = self.extra_params['processor_name']
        else:
            self.processor_name = "microsoft/speecht5_vc"
        if bool(extra_params) and 'model_name' in extra_params:
            self.model_name = self.extra_params['model_name']
        else:
            self.model_name = "microsoft/speecht5_vc"
        if bool(extra_params) and 'vocoder_name' in extra_params:
            self.vocoder_name = self.extra_params['vocoder_name']
        else:
            self.vocoder_name = "microsoft/speecht5_hifigan"
        if bool(extra_params) and 'speaker_embeddings_model_name' in extra_params:
            self.speaker_embeddings_model_name = self.extra_params['speaker_embeddings_model_name']
        else:
            self.speaker_embeddings_model_name = "EcapaTDNN"
        if bool(extra_params) and 'speaker_embeddings_model_checkpoint' in extra_params:
            self.speaker_embeddings_model_checkpoint = self.extra_params['speaker_embeddings_model_checkpoint']
        else:
            self.speaker_embeddings_model_checkpoint = "speechbrain/spkrec-xvect-voxceleb"
         
        # Check if a CUDA-enabled GPU is available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models and processors
        self.processor = SpeechT5Processor.from_pretrained(self.processor_name)
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained(self.model_name).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_name).to(self.device)

        # Initialize the speaker embedding extractor model
        self.speakerEmbeddingExtractor = AudioRepresentation(model_name=self.speaker_embeddings_model_name, 
                                                        model_checkpoint=self.speaker_embeddings_model_checkpoint)

    
    # Define a function for voice anonymization
    def anonymize(self, source_files, target_files, output_files):
        # Loop through pairs of source, target, and output files
        for line in tqdm(zip(source_files, target_files, output_files)):
            source_file, target_file, output_file = line

            # Load source waveform and sample rate
            source_waveform, source_sample_rate = torchaudio.load(source_file)
                        
            with torch.no_grad():
                # Process the source waveform using the SpeechT5 processor
                source_audio_descriptor = self.processor(audio=source_waveform, sampling_rate=source_sample_rate,
                                                    return_tensors="pt")

                # Load target waveform and sample rate
                target_waveform, target_sample_rate = torchaudio.load(target_file)

                # Extract contextual speaker embeddings from the target waveform
                void, target_speaker_embeddings = self.speakerEmbeddingExtractor.contextual_encoding(target_waveform.to(self.device))

                # Generate anonymized voice using the SpeechT5 model and HiFi-GAN vocoder
                output_audio_descriptor = self.model.generate_speech(source_audio_descriptor["input_values"].squeeze(1).to(self.device),
                                                                target_speaker_embeddings.squeeze(1).to(self.device), vocoder=self.vocoder)

            # Save the anonymized voice to the output file
            sf.write(output_file, output_audio_descriptor.cpu().numpy(), samplerate=source_sample_rate)