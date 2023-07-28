# Import necessary libraries
import os
import torch
import librosa
from scipy.io.wavfile import write
from tqdm import tqdm
from pathlib import Path
import sys
import torchaudio

# Set up file paths
freeVC_folder = '../../FreeVC'
script_directory = os.path.dirname(os.path.abspath(__file__))
freeVC_folder_absolute_path = os.path.join(script_directory, freeVC_folder)

# Adding freeVC_folder to the system path
sys.path.insert(0, freeVC_folder_absolute_path)

# Import modules from the freeVC library
import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder


class VoiceAnonymizer:
    def __init__(self, extra_params=None):
        """
        Initialize the VoiceAnonymizer.

        Parameters:
        - extra_params (dict, optional): A dictionary containing additional parameters for customization.
            - 'model_name' (str, optional): The name of the model to use. Defaults to "freevc".
        """
        # Check if the 'model_name' parameter is provided in the extra_params dictionary, otherwise use default
        if bool(extra_params) and 'model_name' in extra_params:
            self.model_name = self.extra_params['model_name']
        else:
            self.model_name = "freevc"   

        # Check if a CUDA-enabled GPU is available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and hyperparameters files
        ptfile = f'{freeVC_folder_absolute_path}/checkpoints/{self.model_name}.pth'
        hpfile = f'{freeVC_folder_absolute_path}/logs/{self.model_name}.json'
        self.hps = utils.get_hparams_from_file(hpfile)

        # Load the SynthesizerTrn model for voice synthesis
        print("Loading model...")
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).to(self.device)
        _ = self.net_g.eval()  # Set the model to evaluation mode
        print("Loading checkpoint...")
        _ = utils.load_checkpoint(ptfile, self.net_g, None, True)

        # Load the WavLM model for content
        print("Loading WavLM for content...")
        self.cmodel = utils.get_cmodel(0)

        # If using a speaker encoder, load the SpeakerEncoder model
        if self.hps.model.use_spk:
            print("Loading speaker encoder...")
            self.smodel = SpeakerEncoder(f'{freeVC_folder_absolute_path}/speaker_encoder/ckpt/pretrained_bak_5805000.pt')


    # Define a function for voice anonymization
    def anonymize(self, source_files, target_files, output_files):
        """
        Perform voice anonymization on a list of source-target audio pairs.

        Parameters:
        - source_files (list of str): List of file paths for the source audio.
        - target_files (list of str): List of file paths for the target audio.
        - output_files (list of str): List of file paths where the converted output will be saved.

        Note:
        The length of source_files, target_files, and output_files should be the same.

        Output:
        - None: The method saves the synthesized audio files to the specified output paths.
        """
        # Process text files and synthesize voice
        print("Processing text...")
        srcs, tgts, outs = source_files, target_files, output_files

        print("Synthesizing...")
        with torch.no_grad():
            for line in tqdm(zip(srcs, tgts, outs)):
                src, tgt, out = line
                # Process target audio file
                wav_tgt, _ = librosa.load(tgt, sr=self.hps.data.sampling_rate)
                wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

                if self.hps.model.use_spk:
                    # Extract speaker embedding if using a speaker encoder
                    g_tgt = self.smodel.embed_utterance(wav_tgt)
                    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(self.device)
                else:
                    # Convert target waveform to Mel spectrogram
                    wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(self.device)
                    mel_tgt = mel_spectrogram_torch(
                        wav_tgt,
                        self.hps.data.filter_length,
                        self.hps.data.n_mel_channels,
                        self.hps.data.sampling_rate,
                        self.hps.data.hop_length,
                        self.hps.data.win_length,
                        self.hps.data.mel_fmin,
                        self.hps.data.mel_fmax
                    )

                # Process source audio file
                wav_src, _ = librosa.load(src, sr=self.hps.data.sampling_rate)
                wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(self.device)
                c = utils.get_content(self.cmodel, wav_src)  # Extract content embedding

                # Perform voice synthesis
                if self.hps.model.use_spk:
                    audio = self.net_g.infer(c, g=g_tgt)  # Use speaker encoder for synthesis
                else:
                    audio = self.net_g.infer(c, mel=mel_tgt)  # Use Mel spectrogram for synthesis
                audio = audio[0][0].data.cpu().float().numpy()

                # Save the synthesized audio to the output file
                Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
                #torchaudio.save(out, audio, self.hps.data.sampling_rate, encoding="PCM_S", bits_per_sample=16)
                write(out, self.hps.data.sampling_rate, audio)
