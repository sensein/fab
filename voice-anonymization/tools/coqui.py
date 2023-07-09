from TTS.api import TTS
import torch
from tqdm import tqdm


def anonymize(source_files, target_files, output_files,
              model_name="voice_conversion_models/multilingual/vctk/freevc24"):
    # Example voice conversion converting speaker of the `source_wav` to the speaker of the `target_wav`
    tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False,
              gpu=torch.cuda.is_available())
    for line in tqdm(zip(source_files, target_files, output_files)):
        source_file, target_file, output_file = line
        tts.voice_conversion_to_file(source_wav=source_file, target_wav=target_file, file_path=output_file)
