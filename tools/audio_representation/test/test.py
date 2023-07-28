import torch
import sys
import os
import warnings

import sys
sys.path.append('../../..')
from tools.audio_representation import AudioRepresentation

# List of representations to test
representations_to_test = [
    {"model_name": "wav2vec2", "model_checkpoint": "facebook/wav2vec2-base-960h", "extra_params": {"layer_number": 0}},
    {"model_name": "wav2vec2", "model_checkpoint": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
     "extra_params": {"layer_number": 0}},
    {"model_name": "wavLM", "model_checkpoint": "microsoft/wavlm-base", "extra_params": {"layer_number": 0}},
    {"model_name": "HuBERT", "model_checkpoint": "facebook/hubert-base-ls960", "extra_params": {"layer_number": 0}},
    {"model_name": "data2vec", "model_checkpoint": "facebook/data2vec-audio-base", "extra_params": {"layer_number": 0}},
    {"model_name": "Whisper", "model_checkpoint": "openai/whisper-base", "extra_params": {"layer_number": 0}},
    {"model_name": "pyannote_audio", "model_checkpoint": None, "extra_params": None},
    {"model_name": "pyannote_audio", "model_checkpoint": None,
     "extra_params": {'hf_token': 'KEY'}},  # TODO: REPLACE HERE
    {"model_name": "pyannote_audio", "model_checkpoint": None, "extra_params": {'window': 'sliding'}},
    {"model_name": "apc", "model_checkpoint": None, "extra_params": {"layer_number": 0}},
    {"model_name": "apc", "model_checkpoint": 'tera_fbankBase_T_F_AdamW_b32_200k_100hr', "extra_params": None},
    {"model_name": "tera", "model_checkpoint": None, "extra_params": {"layer_number": 0}},
    {"model_name": "tera", "model_checkpoint": 'apc_960hr', "extra_params": None},
    {"model_name": "Byol", "model_checkpoint": None, "extra_params": None},
    {"model_name": "EcapaTDNN", "model_checkpoint": None, "extra_params": None},
    {"model_name": "EcapaTDNN", "model_checkpoint": "speechbrain/spkrec-xvect-voxceleb", "extra_params": None},
    {"model_name": "EcapaTDNN", "model_checkpoint": "Ubenwa/ecapa-voxceleb-ft2-cryceleb", "extra_params": None},
    {"model_name": "HumanCochleagram", "model_checkpoint": None, "extra_params": None},
    {"model_name": "LogMelSpectrogram", "model_checkpoint": None, "extra_params": None},
    {"model_name": "FAKE", "model_checkpoint": None, "extra_params": None},
]

#representations_to_test = representations_to_test[:4]
print("representations_to_test")
print(representations_to_test)

# Ignore warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Create a random input waveform
    input_waveform = torch.rand(4, 1, 32000)

    if input_waveform.size(1) == 1:
        input_waveform = input_waveform.squeeze(1)
    else:
        input_waveform = input_waveform[:, :1].squeeze(1)

    print('MONO DATA BATCH')
    print(input_waveform.shape)

    # Iterate over representations to test
    for representation_to_test in representations_to_test:
        print(representation_to_test)

        try:
            # Create an instance of the AudioRepresentation class
            audio_repr = AudioRepresentation(
                model_name=representation_to_test['model_name'],
                model_checkpoint=representation_to_test['model_checkpoint'],
                extra_params=representation_to_test['extra_params']
            )

            try:
                if audio_repr.contextual_encoding_exists:
                    print('CONTEXTUAL AUDIO REPRESENTATION')
                    raw_encoder_response, filtered_encoder_response = audio_repr.contextual_encoding(input_waveform)
                    print(filtered_encoder_response.shape)
                    print(audio_repr.contextual_encoding_size())
                else:
                    print('NO CONTEXTUAL AUDIO REPRESENTATION')
            except Exception as e:
                print("An error has occurred : ", e)

            try:
                if audio_repr.temporal_encoding_exists:
                    print('TEMPORAL AUDIO REPRESENTATION')
                    raw_encoder_response, filtered_encoder_response = audio_repr.temporal_encoding(input_waveform)
                    print(filtered_encoder_response.shape)
                    print(audio_repr.temporal_encoding_size())
                else:
                    print('NO TEMPORAL AUDIO REPRESENTATION')
            except Exception as e:
                print("An error has occurred : ", e)

            try:
                if audio_repr.pooled_temporal_encoding_exists:
                    print('POOLED TEMPORAL AUDIO REPRESENTATION')
                    raw_encoder_response, filtered_encoder_response = audio_repr.pooled_temporal_encoding(input_waveform)
                    print(len(filtered_encoder_response))
                    print(filtered_encoder_response[0]['global_min_pooling'].shape)
                    print(audio_repr.pooled_temporal_encoding_size())
                else:
                    print('NO POOLED TEMPORAL AUDIO REPRESENTATION')
            except Exception as e:
                print("An error has occurred : ", e)
        except Exception as e:
            print("An error has occurred : ", e)

        raw_encoder_response = None
        filtered_encoder_response = None