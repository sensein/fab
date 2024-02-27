from .base_service import BaseService
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
from tqdm.auto import tqdm
import tempfile
import numpy as np

class MMS(BaseService):
    NAME = "mms"

    def __init__(self, configs):
        super().__init__()
        self.model_checkpoint = configs["model_checkpoint"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_cache_dir = self.get_cache_directory()

        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint, 
            cache_dir=self.temp_cache_dir
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_checkpoint,
            cache_dir=self.temp_cache_dir
        ).to(self.device)

    def transcribe(self, input_data):
        # Implementation for MMS
        print("Transcribing with MMS...")
        
        results = []
        raw_results = []
        for input in tqdm(input_data, desc="Transcribing..."):
            audio = np.expand_dims(np.array(input["waveform"]), axis=0)

            sampling_rate = input["sample_rate"]
            language = input.get("language", None)

            if language:
                self.processor.tokenizer.set_target_lang(language)
                self.model.load_adapter(language)

            inputs = self.processor(
                audio, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs).logits

            ids = torch.argmax(outputs, dim=-1)[0]
            transcription = self.processor.decode(ids)
            raw_results.append(transcription)
            results.append({"text": transcription})

        return {
            "formatted": results,
            "raw": raw_results
        }
