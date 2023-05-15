print("this is hpt.py")

import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Python script with encoder and decoder parameters.')

# Add the command-line arguments
parser.add_argument('--encoder', type=str, help='Encoder parameter')
parser.add_argument('--decoder', type=str, help='Decoder parameter')

# Parse the arguments
args = parser.parse_args()

# Access the encoder and decoder values
encoder = args.encoder
decoder = args.decoder

# Print the encoder and decoder values
print(f'Encoder: {encoder}')
print(f'Decoder: {decoder}')

# Your code logic here using the encoder and decoder parameters
# ...
