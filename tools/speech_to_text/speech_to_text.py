# Import Transcriber classes from tools.whisper and tools.mms modules
#from .tools.whisper import Transcriber as Whisper
from .tools.whisper import Transcriber as Whisper
from .tools.mms import Transcriber as MassivelyMultilingualSpeech

# Create a generic Transcriber class that wraps different ASR models based on model_name
class Transcriber:
    def __init__(self, model_name, model_checkpoint=None, language=None, models_save_dir=None, extra_params=None):
        """
        Initialize the Transcriber class.

        Args:
            model_name (str): The name of the ASR model to be used (e.g., 'whisper', 'mms').
            model_checkpoint (str): The checkpoint name for the ASR model. Default is None, which uses a default checkpoint based on the model_name.
            language (str): The target language for transcription. Default is None.
            models_save_dir (str): The directory to save pretrained models. Default is None, which uses a default directory based on the model_name.
            extra_params (dict): Extra parameters for configuration. Default is None.

        Returns:
            None
        """
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint
        self.language = language
        self.models_save_dir = models_save_dir
        self.extra_params = extra_params
        
        # Instantiate the appropriate audio encoder based on the model_name
        if model_name.startswith("whisper"):
            # Create a Whisper ASR model
            self.model = Whisper(model_name=self.model_name, 
                                 model_checkpoint=self.model_checkpoint, 
                                 language=self.language, 
                                 models_save_dir=self.models_save_dir,
                                 extra_params=self.extra_params)
        elif model_name.startswith("mms"):
            # Create a MassivelyMultilingualSpeech ASR model
            self.model = MassivelyMultilingualSpeech(model_name=self.model_name, 
                                 model_checkpoint=self.model_checkpoint, 
                                 language=self.language, 
                                 models_save_dir=self.models_save_dir,
                                 extra_params=self.extra_params)
        else:
            # Raise a ValueError if an invalid model is specified
            raise ValueError("Invalid model specified.")

    def transcribe(self, files):
        """
        Transcribe audio files using the selected ASR model.

        Args:
            files (list): A list of audio file paths.

        Returns:
            tuple: A tuple containing two lists:
                - results (list): A list of dictionaries containing the transcriptions and metadata for each audio file.
                - transcripts (list): A list of strings containing transcriptions for each audio file.
        """
        # Call the transcribe method of the selected ASR model
        results, transcripts = self.model.transcribe(files)
        return results, transcripts
