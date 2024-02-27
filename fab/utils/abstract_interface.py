from jsonschema import validate, ValidationError
import json
import hashlib
import uuid
import os
import functools
from abc import ABC
import time

class AbstractInterface(ABC):
    def __init__(self, 
                 base_dir,
                 input_schema_file='schemas/input.json', 
                 output_schema_file='schemas/output.json'):
        # Default schema paths are set relative to the child's location            
        self.input_schema = self.read_json_schema(os.path.join(base_dir, input_schema_file))
        self.output_schema = self.read_json_schema(os.path.join(base_dir, output_schema_file))

    def schema_validator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Validate input
            input_data = kwargs.get('input_data') or (args[0] if args else {})
            try:
                validate(instance=input_data, schema=self.input_schema)
            except ValidationError as e:
                return f"Input validation error: {e}"

            # Execute the function
            result = func(self, *args, **kwargs)

            # Validate output
            try:
                validate(instance=result, schema=self.output_schema)
            except ValidationError as e:
                return f"Output validation error: {e}"

            return {
                "input": input_data,
                "output": result
            }
        
        return wrapper
    
    @staticmethod
    def read_json_schema(file_path):
        try:
            with open(file_path, 'r') as file:
                schema = json.load(file)
                return schema
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return None
        
    @staticmethod
    def get_service_uuid(service_data):
        # Serialize the JSON object with sorted keys to ensure consistent ordering
        service_data_str = json.dumps(service_data, sort_keys=True)

        # Hash the JSON string using SHA-256
        hash_obj = hashlib.sha256(service_data_str.encode())

        # Use the first 16 bytes of the hash to generate a UUID
        # UUIDs are 128 bits (16 bytes) long
        hash_bytes = hash_obj.digest()[:16]

        # Generate a UUID from the hash bytes
        content_uuid = uuid.UUID(bytes=hash_bytes)

        return content_uuid

    @staticmethod
    def get_response_time(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            result["time"] = {"start": "{:.3f}".format(start_time), 
                              "end": "{:.3f}".format(end_time), 
                              "duration": "{:.3f}".format(duration)}
            return result
        return wrapper