#!/usr/bin/env python
# coding: utf-8


# Let's import some essential libraries that will assist us in our voice anonymization journey.
import os
import pandas as pd
import numpy as np
import torchaudio
import torch
import sys
import pathlib
from speechbrain.utils.metric_stats import EER
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image, Audio
import random
import string
import logging
from jiwer import wer
from statistics import mean, stdev
import json
import ast
import argparse
import contractions
from num2words import num2words
import re

main_folder = "../../.."
sys.path.append(main_folder)
from tools.audio_representation import AudioRepresentation
from tools.voice_anonymization import VoiceAnonymizer
from tools.speech_to_text import Transcriber

# The data_folder variable points to the location where we'll store all the data and audio recordings.
# Think of it as our backstage area, well-organized and ready to showcase the talents of our voices!
data_folder = "../data/"

audio_folder_name = "original_audio_segments"

# target_speakers_for_anonymization_folder represents the folder where we'll keep the chosen speakers.
# These speakers are ready to be anonymized, like a room filled with intriguing characters,
# waiting for their secret identities!
target_speakers_for_anonymization_folder = "target_speakers_for_anonymization/"

# output_folder is where we'll save the output plots and results later on.
output_folder = data_folder + "output/"

# We create the data_folder if it doesn't exist yet, to ensure we have a neat place to store data.
pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

# Suppress debug messages
logging.getLogger('matplotlib.font_manager').disabled = True


# utilities

# Create an AudioRepresentation object, audio_repr, using the "EcapaTDNN" model.
audio_repr = AudioRepresentation(model_name="EcapaTDNN")

# The extract_embeddings function takes an input waveform and returns the corresponding speaker embeddings.
# It uses the audio_repr object to obtain the filtered_encoder_response for the input waveform.
def extract_embeddings(input_waveform):
    raw_encoder_response, filtered_encoder_response = audio_repr.contextual_encoding(input_waveform)
    return filtered_encoder_response

def remove_punctuation_and_lower(text):
    # Convert the text to lowercase
    lower_case_text = text.lower()
    # Remove punctuation
    no_punct_text = ''.join(char for char in lower_case_text if char not in string.punctuation)
    return no_punct_text

# The load_audio function loads an audio file using torchaudio and returns the waveform and sample rate.
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def replace_numbers_with_words(input_str):
    def replace_number(match):
        numeric_part = match.group()
        return num2words(int(numeric_part))

    # Use regular expression to find the numeric part of the input string and replace with words
    result = re.sub(r'\d+', replace_number, input_str)
    return result

def remove_punctuation_and_lower(text):
    # Convert the text to lowercase
    text = text.replace("_", " ")
    text = replace_numbers_with_words(text)
    text = text.lower()    
    text = contractions.fix(text)
    text = text.replace("'", "").strip()  
    text = ''.join(char for char in text if char not in string.punctuation)
    return text

def get_filename(row):
    return os.path.basename(row['path_to_audio_segment_file'])

# The process_files_to_embeddings function processes all the audio files in a given folder to extract speaker embeddings.
# It loads the audio files, groups them by speaker, and then computes the speaker embeddings for each speaker group.
def process_files_to_embeddings_and_transcripts(path_to_audio_folder, handmade_transcript_available=False):
    if handmade_transcript_available:
        xlsx_file_path = os.path.join(path_to_audio_folder, "overall_original.xlsx")
        xlsx_file_path_raw = os.path.join(path_to_audio_folder, "overall_original_raw.xlsx")
    else:
        xlsx_file_path = os.path.join(path_to_audio_folder, "overall_anonymized.xlsx")
        xlsx_file_path_raw = os.path.join(path_to_audio_folder, "overall_anonymized_raw.xlsx")
    
    print("xlsx_file_path")
    print(xlsx_file_path)
    if os.path.exists(xlsx_file_path):
        print('xlsx file already exists')
        
        df = pd.read_excel(xlsx_file_path, index_col=None)
        df['Embeddings'] = df['Embeddings'].apply(ast.literal_eval)
        df.fillna("", inplace=True)
        
        df_raw = pd.read_excel(xlsx_file_path_raw, index_col=None)
        df_raw['Embeddings'] = df_raw['Embeddings'].apply(ast.literal_eval)
        df_raw.fillna("", inplace=True)
    else:
        print('xlsx file does not exist')
        if handmade_transcript_available:
            df = pd.read_excel(os.path.join(path_to_audio_folder, 'segments.xlsx'), engine='openpyxl')
            df['audio_segment_file_base_name'] = df.apply(get_filename, axis=1)

        whisper_transcriber = Transcriber(
            model_name='whisper',
            model_checkpoint=None,
            language="english", 
            models_save_dir=None,
            extra_params=None,
        )

        mms_transcriber = Transcriber(
            model_name='mms',
            model_checkpoint=None,
            language="eng", # TO DO!!!
            models_save_dir=None,
            extra_params=None,
        )
        
        file_details = []
        for file_name in os.listdir(path_to_audio_folder):

            if file_name.endswith(".wav"):
                print(file_name)
                file_path = os.path.join(path_to_audio_folder, file_name)
                waveform, sr = load_audio(file_path)
                speaker = file_name.split("_")[0]

                raw_response, whisper_text = whisper_transcriber.transcribe(waveform)
                
                print("\n")

                if whisper_text is None:
                    whisper_text = ""
                else:
                    whisper_text = remove_punctuation_and_lower(whisper_text[0])
                    #print("whisper_text")
                    print(whisper_text)
                raw_response, mms_text = mms_transcriber.transcribe(waveform)
                if mms_text is None:
                    mms_text = ""
                else:
                    mms_text = remove_punctuation_and_lower(mms_text[0])
                    #print("mms_text")
                    print(mms_text)

                if handmade_transcript_available:
                    transcript = remove_punctuation_and_lower(df.loc[df['audio_segment_file_base_name'] == file_name, 'text'].values[0])
                    print(transcript)
                    file_details.append([file_path, file_name, speaker, waveform.squeeze(), transcript, whisper_text, mms_text])
                else:
                    file_details.append([file_path, file_name, speaker, waveform.squeeze(), whisper_text, mms_text])

        if handmade_transcript_available:
            df = pd.DataFrame(file_details, columns=["Path", "Name", "Speaker", "Waveform", "Handmade transcript", "Whisper transcript", "MMS transcript"])
        else:
            df = pd.DataFrame(file_details, columns=["Path", "Name", "Speaker", "Waveform", "Whisper transcript", "MMS transcript"])
        df_raw = df
        
        if handmade_transcript_available:            
            # Create a new DataFrame with concatenated Waveform values grouped by Speaker
            df_waveform = pd.DataFrame(df.groupby('Speaker', sort=False)['Waveform'].agg(lambda x: np.concatenate(x.values)), columns=['Waveform']).reset_index()

            # Create a new DataFrame with concatenated Transcript values grouped by Speaker
            df_transcript = pd.DataFrame(df.groupby('Speaker', sort=False)['Handmade transcript'].agg(lambda x: " ".join(x)), columns=['Handmade transcript']).reset_index()

            # Create a new DataFrame with concatenated Transcript values grouped by Speaker
            df_transcript_whisper = pd.DataFrame(df.groupby('Speaker', sort=False)['Whisper transcript'].agg(lambda x: " ".join(x)), columns=['Whisper transcript']).reset_index()

            # Create a new DataFrame with concatenated Transcript values grouped by Speaker
            df_transcript_mms = pd.DataFrame(df.groupby('Speaker', sort=False)['MMS transcript'].agg(lambda x: " ".join(x)), columns=['MMS transcript']).reset_index()

            # Merge the two DataFrames on the 'Speaker' column
            df = pd.merge(df_waveform, df_transcript, on='Speaker')
            df = pd.merge(df, df_transcript_whisper, on='Speaker')
            df = pd.merge(df, df_transcript_mms, on='Speaker')
        else:
            df_waveform = pd.DataFrame(df.groupby('Speaker', sort=False)['Waveform'].agg(lambda x: np.concatenate(x.values)), columns=['Waveform']).reset_index()
            # Create a new DataFrame with concatenated Transcript values grouped by Speaker
            df_transcript_whisper = pd.DataFrame(df.groupby('Speaker', sort=False)['Whisper transcript'].agg(lambda x: " ".join(x)), columns=['Whisper transcript']).reset_index()

            # Create a new DataFrame with concatenated Transcript values grouped by Speaker
            df_transcript_mms = pd.DataFrame(df.groupby('Speaker', sort=False)['MMS transcript'].agg(lambda x: " ".join(x)), columns=['MMS transcript']).reset_index()

            # Merge the two DataFrames on the 'Speaker' column
            df = pd.merge(df_waveform, df_transcript_whisper, on='Speaker')
            df = pd.merge(df, df_transcript_mms, on='Speaker')

        all_embeddings = []
        for index, row in df.iterrows():
            waveform = row['Waveform']
            embeddings = extract_embeddings(torch.tensor(waveform))
            all_embeddings.append(embeddings.squeeze())
        
        df['Embeddings'] = [embedding.tolist() for embedding in all_embeddings]
        df['Embeddings'] = df['Embeddings'].apply(json.dumps)
        df.to_excel(xlsx_file_path, index=None)  
        df['Embeddings'] = df['Embeddings'].apply(ast.literal_eval)
        df.fillna("", inplace=True)
        
        all_embeddings = []
        for index, row in df_raw.iterrows():
            waveform = row['Waveform']
            embeddings = extract_embeddings(torch.tensor(waveform))
            all_embeddings.append(embeddings.squeeze())
        
        df_raw['Embeddings'] = [embedding.tolist() for embedding in all_embeddings]
        df_raw['Embeddings'] = df_raw['Embeddings'].apply(json.dumps)
        df_raw.to_excel(xlsx_file_path_raw, index=None)  
        df_raw['Embeddings'] = df_raw['Embeddings'].apply(ast.literal_eval)
        df_raw.fillna("", inplace=True)
    return df, df_raw


# The compute_similarity_score function computes the cosine similarity score between two speaker embeddings.
# It uses the torch.nn.CosineSimilarity to perform the cosine similarity computation.
def compute_similarity_score(embedding1, embedding2):
    cos = torch.nn.CosineSimilarity(dim=-1)
    similarity_score = cos(torch.tensor(embedding1), torch.tensor(embedding2))
    return similarity_score.item()

# The compute_eer_and_plot_verification_scores function compares embeddings and computes the Equal Error Rate (EER).
# It also creates a histogram plot with the verification scores.
def compute_eer_and_plot_verification_scores(df1, df2, output_file_path, seed):
    df_rows = []
    for i, row1 in df1.iterrows():
        for j, row2 in df2.iterrows():
            s1 = row1['Speaker']
            s2 = row2['Speaker']
            e1 = row1['Embeddings']
            e2 = row2['Embeddings']
            cosine = compute_similarity_score(e1, e2)
            if s1 == s2:
                same = 1
            else:
                same = 0
            df_rows.append([s1, s2, e1, e2, cosine, same])

    df_pairs = pd.DataFrame(df_rows, columns=['Speaker 1', 'Speaker 2', 'Embeddings 1', 'Embeddings 2', 'Cosine distance', 'Same'])

    positive_scores = df_pairs.loc[df_pairs['Same'] == True]['Cosine distance'].values
    negative_scores = df_pairs.loc[df_pairs['Same'] == False]['Cosine distance'].values
    eer, threshold = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    plt.figure()
    ax = sns.histplot(df_pairs, x='Cosine distance', hue='Same', stat='percent', common_norm=False)
    ax.set_title(f'EER={round(eer, 4)} - Thresh={round(threshold, 4)}')
    plt.axvline(x=[threshold], color='red', ls='--')
    
    pathlib.Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file_path, format='png')
        # Format the string
    data_string = f"{seed},{eer},{threshold}\n"

    with open(output_file_path.replace(".png", ".csv"), "a") as file:
        file.write(data_string)

    return output_file_path

# The following function, anonymize_audio_files_in_folder, performs the magical voice anonymization process using the coqui voice cloning model.
def anonymize_audio_files_in_folder(tool_name, path_to_audio_folder, path_to_anonymized_audio_folder, target_speaker_file_path):
    # We start by initializing three empty lists to hold information about the audio files.
    source_files = []  # List to store paths of original audio files to be anonymized.
    target_files = []  # List to store paths of the synthetic target speaker's audio file.
    output_files = []  # List to store paths where the anonymized audio files will be saved.

    # Create a VoiceAnonymizer object, named anonymizer, which will perform the voice anonymization.
    anonymizer = VoiceAnonymizer(method=tool_name)

    # Loop through all files in the specified audio folder (path_to_audio_folder).
    for file_name in os.listdir(path_to_audio_folder):
        # Check if the file has a ".wav" extension, indicating an audio file.
        if file_name.endswith(".wav"):
            # Obtain the full path of the audio file.
            file_path = os.path.join(path_to_audio_folder, file_name)

            # Create the corresponding path for the anonymized audio file.
            anonymized_file_path = file_path.replace(path_to_audio_folder, path_to_anonymized_audio_folder)

            # If the anonymized audio file doesn't exist, proceed with the anonymization process.
            if not os.path.exists(anonymized_file_path):
                # Create the necessary directories to store the anonymized audio file.
                pathlib.Path(os.path.dirname(anonymized_file_path)).mkdir(parents=True, exist_ok=True)

                # Add information about the audio files to their respective lists.
                source_files.append(file_path)  # Original audio file to be anonymized.
                target_files.append(target_speaker_file_path)  # Synthetic target speaker's audio file.
                output_files.append(anonymized_file_path)  # Path to save the anonymized audio file.

    # If there are audio files to be anonymized, call the anonymizer.anonymize() method.
    if len(source_files) > 0:
        anonymizer.anonymize(source_files=source_files, target_files=target_files, output_files=output_files)

def extract_detailed_WER(df, anonymized_df):
    h_vs_w__list = []
    h_vs_m__list = []
    h_vs_aw__list = []
    h_vs_am__list = []
    w_vs_aw__list = []
    m_vs_am__list = []
    for i, row in df.iterrows():
        handmade_transcript = row['Handmade transcript']
        whisper_transcript = row['Whisper transcript']
        mms_transcript = row['MMS transcript']
        anon_whisper_transcript = anonymized_df.loc[i, 'Whisper transcript']
        anon_mms_transcript = anonymized_df.loc[i, 'MMS transcript']
        
        '''
        print(handmade_transcript)
        print(whisper_transcript)
        print(mms_transcript)
        print(anon_whisper_transcript)
        print(anon_mms_transcript)
        '''
        
        h_vs_w = wer(handmade_transcript, whisper_transcript)
        h_vs_w__list.append(h_vs_w)
        h_vs_m = wer(handmade_transcript, mms_transcript)
        h_vs_m__list.append(h_vs_m)
        h_vs_aw = wer(handmade_transcript, anon_whisper_transcript)
        h_vs_aw__list.append(h_vs_aw)
        h_vs_am = wer(handmade_transcript, anon_mms_transcript)
        h_vs_am__list.append(h_vs_am)
        if whisper_transcript is None or whisper_transcript == "":
            w_vs_aw = 1 # work around, not really correct
        else:
            w_vs_aw = wer(whisper_transcript, anon_whisper_transcript)
        w_vs_aw__list.append(w_vs_aw)    
        if mms_transcript is None or mms_transcript == "":
            m_vs_am = 1 # work around, not really correct
        else:
            m_vs_am = wer(mms_transcript, anon_mms_transcript)
        m_vs_am__list.append(m_vs_am)
    
    h_vs_w__macro = mean(h_vs_w__list)
    h_vs_m__macro = mean(h_vs_m__list)
    h_vs_aw__macro = mean(h_vs_aw__list)
    h_vs_am__macro = mean(h_vs_am__list)
    w_vs_aw__macro = mean(w_vs_aw__list)
    m_vs_am__macro = mean(m_vs_am__list)
    
    h_vs_w__macro2 = stdev(h_vs_w__list)
    h_vs_m__macro2 = stdev(h_vs_m__list)
    h_vs_aw__macro2 = stdev(h_vs_aw__list)
    h_vs_am__macro2 = stdev(h_vs_am__list)
    w_vs_aw__macro2 = stdev(w_vs_aw__list)
    m_vs_am__macro2 = stdev(m_vs_am__list)
    
    '''
    print(mean(h_vs_w__list))
    print(mean(h_vs_m__list))
    print(mean(h_vs_aw__list))
    print(mean(h_vs_am__list))
    print(mean(w_vs_aw__list))
    print(mean(m_vs_am__list))
    '''
    
    return h_vs_w__list, h_vs_m__list, h_vs_aw__list, h_vs_am__list, w_vs_aw__list, m_vs_am__list, h_vs_w__macro, h_vs_w__macro2, h_vs_m__macro, h_vs_m__macro2, h_vs_aw__macro, h_vs_aw__macro2, h_vs_am__macro, h_vs_am__macro2, w_vs_aw__macro, w_vs_aw__macro2, m_vs_am__macro, m_vs_am__macro2
    
        
def extract_WER(df, anonymized_df, df_raw, anonymized_df_raw, xlsx_file_path1, xlsx_file_path2, xlsx_file_path3):
    pathlib.Path(os.path.dirname(xlsx_file_path1)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(xlsx_file_path2)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(xlsx_file_path2)).mkdir(parents=True, exist_ok=True)

    ### MICRO (one single number describing it all)
    handmade_transcript = " ".join(list(df['Handmade transcript']))
    whisper_transcript = " ".join(list(df['Whisper transcript']))
    mms_transcript = " ".join(list(df['MMS transcript']))
    
    h_vs_w = wer(handmade_transcript, whisper_transcript)
    h_vs_m = wer(handmade_transcript, mms_transcript)

    anon_whisper_transcript = " ".join(list(anonymized_df['Whisper transcript']))
    anon_mms_transcript = " ".join(list(anonymized_df['MMS transcript']))

    h_vs_aw = wer(handmade_transcript, anon_whisper_transcript)
    h_vs_am = wer(handmade_transcript, anon_mms_transcript)
    
    if whisper_transcript is None or whisper_transcript == "":
        w_vs_aw = 1 # work around, not really correct
    else:
        w_vs_aw = wer(whisper_transcript, anon_whisper_transcript)
    if mms_transcript is None or mms_transcript == "":
        m_vs_am = 1 # work around, not really correct
    else:
        m_vs_am = wer(mms_transcript, anon_mms_transcript)
    
    # MACRO 1 (per speaker)
    h_vs_w__list_speaker, h_vs_m__list_speaker, h_vs_aw__list_speaker, h_vs_am__list_speaker, w_vs_aw__list_speaker, m_vs_am__list_speaker, h_vs_w__macro_speaker, h_vs_w__macro2_speaker, h_vs_m__macro_speaker, h_vs_m__macro2_speaker, h_vs_aw__macro_speaker, h_vs_aw__macro2_speaker, h_vs_am__macro_speaker, h_vs_am__macro2_speaker, w_vs_aw__macro_speaker, w_vs_aw__macro2_speaker, m_vs_am__macro_speaker, m_vs_am__macro2_speaker = extract_detailed_WER(df, anonymized_df)
    
    #my_df = pd.DataFrame([[h_vs_w__list_speaker, h_vs_m__list_speaker, h_vs_aw__list_speaker, h_vs_am__list_speaker, w_vs_aw__list_speaker, m_vs_am__list_speaker]], columns=["h_vs_w__list_speaker", "h_vs_m__list_speaker", "h_vs_aw__list_speaker", "h_vs_am__list_speaker", "w_vs_aw__list_speaker", "m_vs_am__list_speaker"])
    data = {
        "h_vs_w__list_speaker": h_vs_w__list_speaker,
        "h_vs_m__list_speaker": h_vs_m__list_speaker,
        "h_vs_aw__list_speaker": h_vs_aw__list_speaker,
        "h_vs_am__list_speaker": h_vs_am__list_speaker,
        "w_vs_aw__list_speaker": w_vs_aw__list_speaker,
        "m_vs_am__list_speaker": m_vs_am__list_speaker
    }
    my_df = pd.DataFrame(data)
    my_df.to_excel(xlsx_file_path1, index=None)  

    # MACRO 2 (per audio)
    h_vs_w__list_audio, h_vs_m__list_audio, h_vs_aw__list_audio, h_vs_am__list_audio, w_vs_aw__list_audio, m_vs_am__list_audio, h_vs_w__macro_audio, h_vs_w__macro2_audio, h_vs_m__macro_audio, h_vs_m__macro2_audio, h_vs_aw__macro_audio, h_vs_aw__macro2_audio, h_vs_am__macro_audio, h_vs_am__macro2_audio, w_vs_aw__macro_audio, w_vs_aw__macro2_audio, m_vs_am__macro_audio, m_vs_am__macro2_audio = extract_detailed_WER(df_raw, anonymized_df_raw)
    #my_df = pd.DataFrame([[h_vs_w__list_audio, h_vs_m__list_audio, h_vs_aw__list_audio, h_vs_am__list_audio, w_vs_aw__list_audio, m_vs_am__list_audio]], columns=["h_vs_w__list_audio", "h_vs_m__list_audio", "h_vs_aw__list_audio", "h_vs_am__list_audio", "w_vs_aw__list_audio", "m_vs_am__list_audio"])
    data = {
        "h_vs_w__list_audio": h_vs_w__list_audio,
        "h_vs_m__list_audio": h_vs_m__list_audio,
        "h_vs_aw__list_audio": h_vs_aw__list_audio,
        "h_vs_am__list_audio": h_vs_am__list_audio,
        "w_vs_aw__list_audio": w_vs_aw__list_audio,
        "m_vs_am__list_audio": m_vs_am__list_audio
    }
    my_df = pd.DataFrame(data) 
    my_df.to_excel(xlsx_file_path2, index=None)  
    
    # OVERALL
    my_df = pd.DataFrame([{"h_vs_w": h_vs_w,
                         "h_vs_m": h_vs_m,
                         "h_vs_aw": h_vs_aw,
                         "h_vs_am": h_vs_am,
                         "w_vs_aw": w_vs_aw,
                         "m_vs_am": m_vs_am,
                         "h_vs_w__macro_speaker": h_vs_w__macro_speaker,
                         "h_vs_w__macro2_speaker": h_vs_w__macro2_speaker,
                         "h_vs_m__macro_speaker": h_vs_m__macro_speaker,
                         "h_vs_m__macro2_speaker": h_vs_m__macro2_speaker,
                         "h_vs_aw__macro_speaker": h_vs_aw__macro_speaker,
                         "h_vs_aw__macro2_speaker": h_vs_aw__macro2_speaker,
                         "h_vs_am__macro_speaker": h_vs_am__macro_speaker,
                         "h_vs_am__macro2_speaker": h_vs_am__macro2_speaker,
                         "w_vs_aw__macro_speaker": w_vs_aw__macro_speaker,
                         "w_vs_aw__macro2_speaker": w_vs_aw__macro2_speaker,
                         "m_vs_am__macro_speaker": m_vs_am__macro_speaker,
                         "m_vs_am__macro2_speaker": m_vs_am__macro2_speaker,
                         "h_vs_w__macro_audio": h_vs_w__macro_audio,
                         "h_vs_w__macro2_audio": h_vs_w__macro2_audio,
                         "h_vs_m__macro_audio": h_vs_m__macro_audio,
                         "h_vs_m__macro2_audio": h_vs_m__macro2_audio,
                         "h_vs_aw__macro_audio": h_vs_aw__macro_audio,
                         "h_vs_aw__macro2_audio": h_vs_aw__macro2_audio,
                         "h_vs_am__macro_audio": h_vs_am__macro_audio,
                         "h_vs_am__macro2_audio": h_vs_am__macro2_audio,
                         "w_vs_aw__macro_audio": w_vs_aw__macro_audio,
                         "w_vs_aw__macro2_audio": w_vs_aw__macro2_audio,
                         "m_vs_am__macro_audio": m_vs_am__macro_audio, 
                         "m_vs_am__macro2_audio": m_vs_am__macro2_audio 
                         }])
    
    my_df.to_excel(xlsx_file_path3, index=None)  
    return my_df

def process_audio_file(file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Check if the audio is stereo, and convert it to mono by taking the left channel
    if waveform.size(0) == 2:
        waveform = waveform[0:1, :]  # Take the left channel

    # Check the sample rate and resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Save the processed waveform as a WAV file with the same file path
    torchaudio.save(file_path, waveform, 16000, encoding="PCM_S", bits_per_sample=16)

    #print(f"Processed audio saved at: {save_path}")

def get_wav_files_in_folder(folder_path):
    wav_files = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(dirpath, filename)
                wav_files.append(file_path)
    return wav_files

def parse_arguments():
    parser = argparse.ArgumentParser(description="Voice Anonymization Script")

    # Add arguments to the parser
    parser.add_argument(
        "--dataset_name", type=str, default="paido_dataset", help="Dataset name"
    )
    parser.add_argument(
        "--anonymization_tool_name",
        type=str,
        default="coqui",
        help="Anonymization tool name",
    )
    parser.add_argument(
        "--target_speaker_for_anonymization_file",
        type=str,
        #default="fem_child_synthetic_stan.wav",
        default="timmy_child_narakeet.wav",
        help="Target speaker filename",
    )
    # Add arguments to the parser
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed number"
    )

    # Add other arguments if needed

    return parser.parse_args()
        
def main():
    args = parse_arguments()

    # Accessing the values from the parsed arguments
    dataset_name = args.dataset_name
    anonymization_tool_name = args.anonymization_tool_name
    target_speaker_for_anonymization_file = args.target_speaker_for_anonymization_file
    seed = args.seed

    # Next, we start initializing some variables.
    # target_speaker_file_path points to the audio file of our chosen speaker for anonymization.
    target_speaker_file_path = data_folder + target_speakers_for_anonymization_folder + target_speaker_for_anonymization_file
        # dataset_path holds the path to our downloaded EmoDB dataset.
    dataset_path = data_folder + dataset_name
    # audio_folder_path points to the folder containing the original audio recordings in the EmoDB dataset.
    audio_folder_path = f"{dataset_path}/{audio_folder_name}"
    # anonymized_audio_folder_path is where we'll save the anonymized audio recordings.
    # We'll create a new folder specific to the anonymization tool used for clarity.
    anonymized_audio_folder_path = dataset_path + f"/anonymized_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{audio_folder_name}_{seed}/"

    # Print information about the chosen dataset and anonymization tool
    print("Dataset Name:", dataset_name)
    print("Anonymization Tool:", anonymization_tool_name)
    print("Target Speaker for Anonymization:", target_speaker_for_anonymization_file)
    print("Seed:", seed)

    # The following code invokes the function anonymize_audio_files_in_folder to perform the voice anonymization.

    # The function is called with the following arguments:
    # - anonymization_tool_name: The name of the tool to use for anonymization (in this case, "coqui").
    # - audio_folder_path: The path to the folder containing the original audio recordings in the EmoDB dataset.
    # - anonymized_audio_folder_path: The path where the anonymized audio recordings will be saved.
    # - target_speaker_file_path: The path to the audio file of our chosen target speaker ("matt_male_adult_narakeet.wav").
    print("\nStarting voice anonymization process...")
    anonymize_audio_files_in_folder(anonymization_tool_name, audio_folder_path, anonymized_audio_folder_path, target_speaker_file_path)
    wav_files = get_wav_files_in_folder(anonymized_audio_folder_path)
    for file in wav_files:
        process_audio_file(file)    
    print("Voice anonymization process completed.")

    # ### Evaluating voice anonymization based on the achieved privacy and utility

    # The following code processes the audio files in the specified audio_folder_path
    # and extracts speaker embeddings for each speaker.
    print("\nProcessing audio files to obtain speaker embeddings...")
    df, df_raw = process_files_to_embeddings_and_transcripts(audio_folder_path, handmade_transcript_available=True)
    print("Speaker embeddings extraction for original audio completed.")

    # Calling the process_files_to_embeddings function with the anonymized_audio_folder_path as input.
    # This function will process the audio files in the anonymized_audio_folder_path and extract speaker embeddings for each speaker.
    anonymized_df, anonymized_df_raw = process_files_to_embeddings_and_transcripts(anonymized_audio_folder_path)
    print("Speaker embeddings extraction for anonymized audio completed.")

    # Print information about the evaluation phase
    print("\nEvaluating voice anonymization based on the achieved privacy and utility...")

    # The plot_path variable will store the path to the generated plot.
    plot_path = compute_eer_and_plot_verification_scores(df, df, f'{output_folder}{dataset_name}/EER_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{dataset_name}_original_original.png', seed)
    print("Verification scores computed for original audio.")

    # The plot_path variable will store the path to the generated plot.
    plot_path = compute_eer_and_plot_verification_scores(df_raw, df_raw, f'{output_folder}{dataset_name}/EER_raw_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{dataset_name}_original_original.png', seed)
    print("Verification scores computed for original audio.")
    
    # The plot_path variable will store the path to the generated plot.
    plot_path = compute_eer_and_plot_verification_scores(df, anonymized_df, f'{output_folder}{dataset_name}/EER_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{dataset_name}_{seed}_original_anonymized.png', seed)
    print("Verification scores computed for original vs. anonymized audio.")

    # The plot_path variable will store the path to the generated plot.
    plot_path = compute_eer_and_plot_verification_scores(df_raw, anonymized_df_raw, f'{output_folder}{dataset_name}/EER_raw_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{dataset_name}_{seed}_original_anonymized.png', seed)
    print("Verification scores computed for original vs. anonymized audio.")

    # The plot_path variable will store the path to the generated plot.
    plot_path = compute_eer_and_plot_verification_scores(anonymized_df, anonymized_df, f'{output_folder}{dataset_name}/EER_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{dataset_name}_{seed}_anonymized_anonymized.png', seed)
    print("Verification scores computed for anonymized audio.")

    # The plot_path variable will store the path to the generated plot.
    plot_path = compute_eer_and_plot_verification_scores(anonymized_df_raw, anonymized_df_raw, f'{output_folder}{dataset_name}/EER_raw_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{dataset_name}_{seed}_anonymized_anonymized.png', seed)
    print("Verification scores computed for anonymized audio.")

    # Perform Word Error Rate (WER) evaluation
    extract_WER(df, anonymized_df, df_raw, anonymized_df_raw, f'{output_folder}{dataset_name}/WER_speaker_details, {dataset_name}_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{seed}.xlsx', f'{output_folder}{dataset_name}/WER_audio_details_{dataset_name}_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{seed}.xlsx', f'{output_folder}{dataset_name}/WER_overall_{dataset_name}_{anonymization_tool_name}_{target_speaker_for_anonymization_file[:-4]}_{seed}.xlsx')
    print("Word Error Rate (WER) evaluation completed.")
    
if __name__ == "__main__":
    main()