from .base_service import BaseService
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np
from tqdm.auto import tqdm
import tempfile

class Whisper(BaseService):
    NAME = "whisper"

    def __init__(self, configs):
        super().__init__()

        self.model_checkpoint = configs["model_checkpoint"]
        self.batch_size = configs["batch_size"] or 16
        self.chunk_length_s = configs["chunk_length_s"] or 30 
        self.max_new_tokens = configs["max_new_tokens"] or 128
        self.return_timestamps = configs["return_timestamps"] or False

        self.temp_cache_dir = self.get_cache_directory()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_checkpoint, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            cache_dir=self.temp_cache_dir
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint,
            cache_dir=self.temp_cache_dir
)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=self.max_new_tokens,
            chunk_length_s=self.chunk_length_s, 
            batch_size=self.batch_size,
            return_timestamps=self.return_timestamps,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe(self, input_data):
        results = []
        raw_results = []

        for input in tqdm(input_data, desc="Transcribing..."):
            audio_waveform = np.array(input["waveform"])
            language = input.get('language', None)
            generate_kwargs = {"language": language} if language else {}
        
            result = self.pipe(audio_waveform, generate_kwargs=generate_kwargs)

            raw_results.append(result)
            for chunk in result['chunks']:
                chunk['timestamps'] = list(chunk['timestamp'])
            results.append(result)

        return {
            "formatted": results,
            "raw": raw_results
        }
