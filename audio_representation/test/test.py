import torch
import sys
import os
import warnings

target_folder = '../'
script_directory = os.path.dirname(os.path.abspath(__file__))
target_folder_absolute_path = os.path.join(script_directory, target_folder)
# adding target_folder to the system path
sys.path.insert(0, target_folder_absolute_path)

from audio_representation import AudioRepresentation

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    input_waveform = torch.rand(4, 1, 32000)

    if input_waveform.size(1) == 1:
        input_waveform = input_waveform.squeeze(1)
    else:
        input_waveform = input_waveform[:, :1].squeeze(1)

    print('MONO DATA BATCH')
    print(input_waveform.shape)

    audio_repr = AudioRepresentation(model_name="byol", model_checkpoint=None, extra_params=None)  # SpkrecEcapaVoxceleb, FacebookModel, HumanCochleagram, LogMelSpectrogram

    if audio_repr.contextual_encoding_exists:
        print('CONTEXTUAL AUDIO REPRESENTATION')
        encoded_output = audio_repr.contextual_encoding(input_waveform)
        print(encoded_output.shape)
        print(audio_repr.contextual_encoding_size())
    else:
        print('NO CONTEXTUAL AUDIO REPRESENTATION')

    if audio_repr.temporal_encoding_exists:
        print('TEMPORAL AUDIO REPRESENTATION')
        encoded_output = audio_repr.temporal_encoding(input_waveform)
        print(encoded_output.shape)
        print(audio_repr.temporal_encoding_size())
    else:
        print('NO TEMPORAL AUDIO REPRESENTATION')

    if audio_repr.pooled_temporal_encoding_exists:
        print('POOLED TEMPORAL AUDIO REPRESENTATION')
        encoded_output = audio_repr.pooled_temporal_encoding(input_waveform)
        print(len(encoded_output))
        print(encoded_output[0]['global_min_pooling'].shape)
        print(audio_repr.pooled_temporal_encoding_size())
    else:
        print('NO POOLED TEMPORAL AUDIO REPRESENTATION')
