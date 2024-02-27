from fab.tasks import SpeechToText

import numpy as np
tensor1 = np.random.rand(32000).tolist()
tensor2 = np.random.rand(32000).tolist()
tensor3 = np.random.rand(32000).tolist()

# Creating an instance and calling the method
speech_to_text = SpeechToText()
result = speech_to_text.transcribe(
    {
    "input": [
        {
        "waveform": tensor1,
        "sample_rate": 16000,
        },
        {
        "waveform": tensor2,
        "sample_rate": 16000,
        },
        {
        "waveform": tensor3,
        "sample_rate": 16000,
        }
    ],
    "service": {
        "service_name": "whisper", 
        "model_checkpoint": "openai/whisper-tiny",
        "batch_size": 16,
        "chunk_length_s": 30,
        "max_new_tokens": 128,
        "return_timestamps": "word"
        }
    }
)
print(result['output']['formatted'])
print(result['time'])

"""
result = speech_to_text.transcribe(
    {
    "input": [
        {
        "waveform": tensor1,
        "sample_rate": 16000,
        "language": "eng",
        },
        {
        "waveform": tensor2,
        "sample_rate": 16000,
        "language": "eng",
        },
        {
        "waveform": tensor3,
        "sample_rate": 16000,
        }
    ],
    "service": {
        "service_name": "mms", 
        "model_checkpoint": "facebook/mms-1b-all",
        }
    }
)
print(result['output']['formatted'])
print(result['time'])
"""
result = speech_to_text.transcribe(
    {
    "input": [
        {
        "waveform": tensor1,
        "sample_rate": 16000,
        "language": "en",
        },
        {
        "waveform": tensor2,
        "sample_rate": 16000,
        },
        {
        "waveform": tensor3,
        "sample_rate": 16000,
        }
    ],
    "service": {
        "service_name": "faster_whisper", 
        "model_checkpoint": "tiny",
        "batch_size": 16,
        }
    }
)
print(result['output']['formatted'])
print(result['time'])