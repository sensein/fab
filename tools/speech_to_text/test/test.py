import torch
import sys
import os
import warnings

sys.path.append('../../..')
from tools.speech_to_text import Transcriber

# List of representations to test
models_to_test = [
    {"model_name": "mms", "model_checkpoint": None, "language": None, "models_save_dir": None, "extra_params": None},
    {"model_name": "whisper", "model_checkpoint": None, "language": None, "models_save_dir": None, "extra_params": None},
    {"model_name": "whisper", "model_checkpoint": None, "language": None, "models_save_dir": None, "extra_params": {"word_timestamps":True}},
]

print("models_to_test")
print(models_to_test)

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
    for model_to_test in models_to_test:
        print(model_to_test)

        try:
            # Create an instance of the AudioRepresentation class
            transcriber = Transcriber(
                model_name=model_to_test['model_name'],
                model_checkpoint=model_to_test['model_checkpoint'],
                language=model_to_test['language'],
                models_save_dir=model_to_test['models_save_dir'],
                extra_params=model_to_test['extra_params'],
            )

            raw_response, text = transcriber.transcribe(input_waveform)
            print(raw_response)
            print(text)
        except Exception as e:
            print("An error has occurred : ", e)