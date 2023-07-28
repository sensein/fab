# Import the necessary anonymization methods from different modules
from .tools.coqui import VoiceAnonymizer as CoquiAnonimizer
from .tools.freevc import VoiceAnonymizer as FreeVCAnonimizer
from .tools.mcadams import VoiceAnonymizer as McAdamsAnonimizer
from .tools.speechT5 import VoiceAnonymizer as SpeechT5Anonimizer
#from .tools.phonetic_intermediate_representations import VoiceAnonymizer as PirAnonimizer


# Define the VoiceAnonymizer class
class VoiceAnonymizer:
    def __init__(self, method, extra_params=None):
        # Define the available anonymization methods
        self.ANONYMIZATION_METHODS = {
            'COQUI': 'coqui',
            'FREEVC': 'freevc',
            'MCADAMS': 'mcadams',
            'SPEECHT5': 'speechT5',
            'PIR': 'pir',
        }

        # Define the methods that require target_files during anonymization
        self.METHODS_WITH_TARGET = {
            self.ANONYMIZATION_METHODS['COQUI'],
            self.ANONYMIZATION_METHODS['FREEVC'],
            self.ANONYMIZATION_METHODS['SPEECHT5']
        }

        # Define the methods that don't require target_files
        self.METHODS_WITHOUT_TARGET = {
            self.ANONYMIZATION_METHODS['MCADAMS'],
            self.ANONYMIZATION_METHODS['PIR'],
        }
        
        self.method = method
        
        # Validate the input parameters
        # Check if the given method is valid
        if self.method not in self.ANONYMIZATION_METHODS.values():
            raise ValueError(f"Invalid anonymization method: {self.method}")

        # Perform anonymization based on the given method
        # Anonymize using the COQUI method
        if self.method == self.ANONYMIZATION_METHODS['COQUI']:
            self.anonymizer = CoquiAnonimizer(extra_params=extra_params)
        # Anonymize using the SPEECHT5 method
        elif self.method == self.ANONYMIZATION_METHODS['SPEECHT5']:
            self.anonymizer = SpeechT5Anonimizer(extra_params=extra_params)
        # Anonymize using the FREEVC method
        elif self.method == self.ANONYMIZATION_METHODS['FREEVC']:
            self.anonymizer = FreeVCAnonimizer(extra_params=extra_params)
        # Anonymize using the MCADAMS method
        elif self.method == self.ANONYMIZATION_METHODS['MCADAMS']:
            self.anonymizer = McAdamsAnonimizer(extra_params=extra_params)
        # Raise an error if the given method is invalid
        else:
            raise ValueError(f"Invalid anonymization method: {self.method}")

        '''
        # Anonymize using the PIR method
        elif self.method == self.ANONYMIZATION_METHODS['PIR']:
            self.anonymizer = PirAnonimizer(extra_params=extra_params)
        '''
            
    # Anonymize method for performing the actual anonymization
    def anonymize(self, source_files, target_files=None, output_files=None):
        # Validate the input parameters

        # Check if source_files is provided and not empty
        if source_files is None or len(source_files) <= 0:
            raise ValueError("source_files cannot be None or empty.")

        # Check if target_files is required for the given method and if it is provided and not empty
        if self.method in self.METHODS_WITH_TARGET and (target_files is None or len(target_files) <= 0):
            raise ValueError(f"if method == {self.method}, target_files cannot be None or empty.")

        # Check if output_files is provided and not empty
        if output_files is None or len(output_files) <= 0:
            raise ValueError("output_files cannot be None or empty.")

        # Check if source_files and target_files have the same length (if target_files is provided)
        if target_files and len(source_files) != len(target_files):
            raise ValueError("source_files and target_files must have the same length.")

        # Check if source_files and output_files have the same length
        if len(source_files) != len(output_files):
            raise ValueError("source_files and output_files must have the same length.")

        # Perform anonymization based on the given method
        # Anonymize using the COQUI method
        if self.method == self.ANONYMIZATION_METHODS['COQUI']:
            self.anonymizer.anonymize(source_files=source_files, target_files=target_files, output_files=output_files)

        # Anonymize using the FREEVC method
        elif self.method == self.ANONYMIZATION_METHODS['FREEVC']:
            self.anonymizer.anonymize(source_files=source_files, target_files=target_files, output_files=output_files)

        # Anonymize using the SPEECHT5 method
        elif self.method == self.ANONYMIZATION_METHODS['SPEECHT5']:
            self.anonymizer.anonymize(source_files=source_files, target_files=target_files, output_files=output_files)
        
        # Anonymize using the MCADAMS method
        elif self.method == self.ANONYMIZATION_METHODS['MCADAMS']:
            self.anonymizer.anonymize(source_files=source_files, output_files=output_files)

        # Raise an error if the given method is invalid
        else:
            raise ValueError(f"Invalid anonymization method: {self.method}")
            
        '''
        # Anonymize using the PIR method
        elif self.method == self.ANONYMIZATION_METHODS['PIR']:
            self.anonymizer.anonymize(source_files=source_files, output_files=output_files)
        '''