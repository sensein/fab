# Import the necessary anonymization methods from different modules
from tools.coqui import anonymize as anonymize_coqui
from tools.freevc import anonymize as anonymize_freevc
from tools.mcadams import anonymize as anonymize_mcadams
from tools.speechT5 import anonymize as anonymize_speechT5


# Define the VoiceAnonymizer class
class VoiceAnonymizer:
    def __init__(self):
        # Define the available anonymization methods
        self.ANONYMIZATION_METHODS = {
            'COQUI': 'coqui',
            'FREEVC': 'freevc',
            'MCADAMS': 'mcadams',
            'SPEECHT5': 'speechT5'
        }

        # Define the methods that require target_files during anonymization
        self.METHODS_WITH_TARGET = {
            self.ANONYMIZATION_METHODS['COQUI'],
            self.ANONYMIZATION_METHODS['FREEVC'],
            self.ANONYMIZATION_METHODS['SPEECHT5']
        }

        # Define the method that doesn't require target_files
        self.METHODS_WITHOUT_TARGET = {
            self.ANONYMIZATION_METHODS['MCADAMS']
        }

    # Anonymize method for performing the actual anonymization
    def anonymize(self, method, source_files, target_files=None, output_files=None):
        # Validate the input parameters

        # Check if the given method is valid
        if method not in self.ANONYMIZATION_METHODS.values():
            raise ValueError(f"Invalid anonymization method: {method}")

        # Check if source_files is provided and not empty
        if source_files is None or len(source_files) <= 0:
            raise ValueError("source_files cannot be None or empty.")

        # Check if target_files is required for the given method and if it is provided and not empty
        if method in self.METHODS_WITH_TARGET and (target_files is None or len(target_files) <= 0):
            raise ValueError(f"if method == {method}, target_files cannot be None or empty.")

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
        if method == self.ANONYMIZATION_METHODS['COQUI']:
            anonymize_coqui(source_files=source_files, target_files=target_files, output_files=output_files)

        # Anonymize using the FREEVC method
        elif method == self.ANONYMIZATION_METHODS['FREEVC']:
            anonymize_freevc(source_files=source_files, target_files=target_files, output_files=output_files)

        # Anonymize using the FREEVC method
        elif method == self.ANONYMIZATION_METHODS['SPEECHT5']:
            anonymize_speechT5(source_files=source_files, target_files=target_files, output_files=output_files)

        # Anonymize using the MCADAMS method
        elif method == self.ANONYMIZATION_METHODS['MCADAMS']:
            anonymize_mcadams(source_files=source_files, output_files=output_files)

        # Raise an error if the given method is invalid
        else:
            raise ValueError(f"Invalid anonymization method: {method}")