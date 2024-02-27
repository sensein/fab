from .base_service import BaseService
from tqdm.auto import tqdm
import torch 
#import whisperx
import numpy as np
from faster_whisper import WhisperModel

class FasterWhisper(BaseService):
    NAME = "faster_whisper"

    def __init__(self, configs):
        super().__init__()

        self.model_checkpoint = configs["model_checkpoint"]
        self.batch_size = configs.get("batch_size", 16)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = "float16" if self.device == "cuda" else "float32"

        self.temp_cache_dir = self.get_cache_directory()

        self.model = WhisperModel(self.model_checkpoint,
                                  device=self.device,
                                  compute_type=self.torch_dtype,
                                  download_root=self.temp_cache_dir)
        
        self.return_timestamps=configs.get("return_timestamps", False)

    def transcribe(self, input_data):
        formatted_results, raw_results = [], []

        for input_item in tqdm(input_data, desc="Transcribing..."):
            audio_waveform = np.array(input_item["waveform"])
            language = input_item.get('language', None)
            segments, info = self.model.transcribe(audio_waveform, 
                                                   beam_size=5, 
                                                   language=language)

            if self.return_timestamps == "word":
                my_segments = []
                if len(list(segments)) > 0:
                    for segment in segments:
                        for segment in segment.words:
                            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.word))
                            my_segments.append({
                                "text": segment.word,
                                "timestamps": [segment.start, segment.end]
                            })
                text = " ".join(segment['text'] for segment in my_segments)
                chunks = [{"text": segment['text'], "timestamps": [segment['start'], segment['end']]} for segment in my_segments]
            else:
                my_segments = [{
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                } for segment in segments]

                raw_results.append({
                    "segments": my_segments,
                    "info": info
                })

                text = "".join(segment['text'] for segment in my_segments)                    
                chunks = [{"text": segment['text'], "timestamps": [segment['start'], segment['end']]} for segment in my_segments]
            formatted_results.append({"text": text, "chunks": chunks})
        
        return {
            "formatted": formatted_results,
            "raw": raw_results
        }
    







