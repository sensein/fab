import torch
from audio_representation import AudioRepresentation

audio_repr = AudioRepresentation(model_name="SpkrecEcapaVoxceleb")
input_waveform = torch.rand(1, 16000)
encoded_output = audio_repr.contextual_encoding(input_waveform)
print(encoded_output)