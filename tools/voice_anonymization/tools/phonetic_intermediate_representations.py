import os
import numpy as np
import torch
import torchaudio
import sys
import audiosegment
from pathlib import Path
from scipy.io import wavfile

# Set up file paths
pir_folder = '../../pir'
script_directory = os.path.dirname(os.path.abspath(__file__))
pir_folder_absolute_path = os.path.join(script_directory, pir_folder)
# Adding freeVC_folder to the system path
sys.path.insert(0, pir_folder_absolute_path)

from demo_inference.demo_tts import DemoTTS
from demo_inference.demo_asr import DemoASR
from demo_inference.demo_anonymization import DemoAnonymizer


class VoiceAnonymizer:
    def __init__(self, extra_params=None):    
        self.extra_params = extra_params
        if bool(extra_params) and 'anon_tag' in extra_params:
            self.anon_tag = self.extra_params['anon_tag']
        else:
            self.anon_tag = "pool"
        if bool(extra_params) and 'asr_tag' in extra_params:
            self.asr_tag = self.asr_tag['processor_name']
        else:
            self.asr_tag = "phones"
        if bool(extra_params) and 'tts_tag' in extra_params:
            self.tts_tag = self.extra_params['tts_tag']
        else:
            self.tts_tag = "Libri100"

        # Check if a CUDA-enabled GPU is available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.path_to_anon_models = Path(pir_folder_absolute_path, "models", "anonymization")
        self.path_to_asr_models = Path(pir_folder_absolute_path, "models", "asr")
        self.path_to_tts_models = Path(pir_folder_absolute_path, "models", "tts")

        self.synthesis_model = DemoTTS(
            model_paths=self.path_to_tts_models, 
            model_tag=self.tts_tag, 
            device=self.device
        )

        self.asr_model = DemoASR(
            model_path=self.path_to_asr_models,
            model_tag=self.asr_tag, 
            device=self.device
        )

        self.anon_model = DemoAnonymizer(
            model_path=self.path_to_anon_models, 
            model_tag=self.anon_tag,
            device=self.device
        )
            
    def anonymize_recording(recording):
        sr, audio = recording
        audio = self.pcm2float(audio)

        text = self.asr_model.recognize_speech(audio, sr)
        speaker_embedding = self.anon_model.anonymize_embedding(audio, sr)

        syn_audio = self.synthesis_model.read_text(
            transcription = text,
            speaker_embedding=speaker_embedding,
            text_is_phonemes= self.asr_tag == "phones"
        )
        return 48000, float2pcm(syn_audio.cpu().numpy()) # TODO! OK???


    def anonymize(self, source_files, output_files):
        """
        Assuming --source_files-- represents a director of audio files to be anonymized and --output_files-- 
        is where the anonymized files should be placed

        anon_tag can be one of ['pool', 'random', 'pool raw']
        tts_tag can be one of ['Libri100', 'Libri100 + finetuned', 'Libri600', 'Libri600 + finetuned']
        asr_tag can be one of ['phones', 'STT', 'TTS']
        """                           
        for i, path_to_file in source_files:        
            input_sound = audiosegment.from_file(path_to_file)

            #Padding audio if too short
            if input_sound.duration_seconds < 2:
                silence = audiosegment.silent(duration = 2000 - len(input_sound) + 1)
                padded_input = input_sound + silence
                rate = padded_input.frame_rate
                data = padded_input.to_numpy_array()
            else:
                rate, data = wavfile.read(path_to_file)

            input_data = (rate, data)

            #Try-except clauses to ignore empty sounds in directory            
            try:
                sr, anonymized_wav = self.anonymize_recording(input_data)
            except RuntimeError as e:
                if str(e) == "mat1 and mat2 shapes cannot be multiplied (1x0 and 66x100)":
                    continue
                else:
                    raise e

            path_to_anonymized_file = output_files[i]
            Path(os.path.dirname(path_to_anonymized_file)).mkdir(parents=True, exist_ok=True)
            wavfile.write(path_to_anonymized_file, sr, anonymized_wav)

    def pcm2float(self, sig, dtype='float32'):
        """
        https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
        """
        sig = np.asarray(sig)
        if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max


    def float2pcm(self, sig, dtype='int16'):
        """
        https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
        """
        sig = np.asarray(sig)
        if sig.dtype.kind != 'f':
            raise TypeError("'sig' must be a float array")
        dtype = np.dtype(dtype)
        if dtype.kind not in 'iu':
            raise TypeError("'dtype' must be an integer type")
        i = np.iinfo(dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)