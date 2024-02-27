from abc import abstractmethod
from fab.utils import VeryBaseService

class BaseService(VeryBaseService):
    def __init__(self):
        pass

    @abstractmethod
    def transcribe(self, input_data):
        # Subclasses must implement this method
        pass