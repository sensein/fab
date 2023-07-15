# Import necessary libraries
import os
import torch
import librosa
from scipy.io.wavfile import write
from tqdm import tqdm
from pathlib import Path
import sys

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

# Define a function for voice anonymization
def anonymize(source_files, target_files, output_files, model_name="freevc"):
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and hyperparameters files
    ptfile = f'{freeVC_folder_absolute_path}/checkpoints/{model_name}.pth'
    hpfile = f'{freeVC_folder_absolute_path}/logs/{model_name}.json'
    hps = utils.get_hparams_from_file(hpfile)

    # Load the SynthesizerTrn model for voice synthesis
    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    _ = net_g.eval()  # Set the model to evaluation mode
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(ptfile, net_g, None, True)

    # Load the WavLM model for content
    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    # If using a speaker encoder, load the SpeakerEncoder model
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder(f'{freeVC_folder_absolute_path}/speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    # Process text files and synthesize voice
    print("Processing text...")
    srcs, tgts, outs = source_files, target_files, output_files

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(srcs, tgts, outs)):
            src, tgt, out = line
            # Process target audio file
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

            if hps.model.use_spk:
                # Extract speaker embedding if using a speaker encoder
                g_tgt = smodel.embed_utterance(wav_tgt)
                g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
            else:
                # Convert target waveform to Mel spectrogram
                wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(device)
                mel_tgt = mel_spectrogram_torch(
                    wav_tgt,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )

            # Process source audio file
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(device)
            c = utils.get_content(cmodel, wav_src)  # Extract content embedding

            # Perform voice synthesis
            if hps.model.use_spk:
                audio = net_g.infer(c, g=g_tgt)  # Use speaker encoder for synthesis
            else:
                audio = net_g.infer(c, mel=mel_tgt)  # Use Mel spectrogram for synthesis
            audio = audio[0][0].data.cpu().float().numpy()

            # Save the synthesized audio to the output file
            Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
            write(out, hps.data.sampling_rate, audio)
