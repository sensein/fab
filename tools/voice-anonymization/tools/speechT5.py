# Import necessary libraries
import sys
import os
import soundfile as sf
import torchaudio
from tqdm import tqdm

# Set up file paths
audio_representation_folder = '../../audio_representation/'
script_directory = os.path.dirname(os.path.abspath(__file__))
audio_representation_folder_absolute_path = os.path.join(script_directory, audio_representation_folder)

# Add audio_representation_folder to the system path
sys.path.insert(0, audio_representation_folder_absolute_path)

# Import modules from the audio_representation library
from audio_representation import AudioRepresentation

# Import models from the transformers library
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan

# Initialize models and processors
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Initialize the speaker embedding extractor model
speakerEmbeddingExtractor = AudioRepresentation(model_name="SpkrecEcapaVoxceleb",
                                                model_checkpoint="speechbrain/spkrec-xvect-voxceleb")


# Define a function for voice anonymization
def anonymize(source_files, target_files, output_files):
    # Loop through pairs of source, target, and output files
    for line in tqdm(zip(source_files, target_files, output_files)):
        source_file, target_file, output_file = line

        # Load source waveform and sample rate
        source_waveform, source_sample_rate = torchaudio.load(source_file)

        # Process the source waveform using the SpeechT5 processor
        source_audio_descriptor = processor(audio=source_waveform, sampling_rate=source_sample_rate,
                                            return_tensors="pt")

        # Load target waveform and sample rate
        target_waveform, target_sample_rate = torchaudio.load(target_file)

        # Extract contextual speaker embeddings from the target waveform
        target_speaker_embeddings = speakerEmbeddingExtractor.contextual_encoding(target_waveform)

        # Generate anonymized voice using the SpeechT5 model and HiFi-GAN vocoder
        output_audio_descriptor = model.generate_speech(source_audio_descriptor["input_values"].squeeze(1),
                                                        target_speaker_embeddings.squeeze(1), vocoder=vocoder)

        # Save the anonymized voice to the output file
        sf.write(output_file, output_audio_descriptor.numpy(), samplerate=source_sample_rate)
