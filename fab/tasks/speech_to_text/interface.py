from fab.utils import AbstractInterface
from .services import Whisper
from .services import MMS
from .services import FasterWhisper
import os

class SpeechToText(AbstractInterface):
    # this class works as a factory for service instances

    _instances = {}
    """
    Note:
    - Instance Method: If _instances should be unique to each instance of SpeechToText.
    - Class Attribute: If _instances is intended to be a shared cache across all instances.
    """

    def __init__(self):
        super().__init__(os.path.dirname(__file__))
        
    @classmethod
    def get_service(cls, service_data):
        # Use a composite key to uniquely identify instances
        key = cls.get_service_uuid(service_data)   

        if key not in cls._instances:
            if service_data["service_name"] == Whisper.NAME:
                cls._instances[key] = Whisper(service_data)
            elif service_data["service_name"] == MMS.NAME:
                cls._instances[key] = MMS(service_data)
            elif service_data["service_name"] == FasterWhisper.NAME:
                cls._instances[key] = FasterWhisper(service_data)
            else:
                raise ValueError(f"Unsupported service: {service_data['service_name']}")
        return cls._instances[key]

    @AbstractInterface.get_response_time
    @AbstractInterface.schema_validator
    def transcribe(self, input_data):
        service = self.get_service(input_data["service"])
        return service.transcribe(input_data["input"])