from tqdm import tqdm
import os
import pathlib
import torchaudio
import random
import numpy as np
import scipy.io.wavfile
import librosa
import torch


class VoiceAnonymizer:
    def __init__(self, extra_params=None):
        """
        Initialize the VoiceAnonymizer.

        Parameters:
        - extra_params (dict, optional): A dictionary containing additional parameters for customization.
            - 'random_coef_min' (float, optional): The minimum value for the random coefficient used in anonymization.
                                                   Defaults to 0.5.
            - 'random_coef_max' (float, optional): The maximum value for the random coefficient used in anonymization.
                                                   Defaults to 0.9.
            - 'winLengthinms' (int, optional): Analysis window length in milliseconds. Defaults to 20 ms.
            - 'shiftLengthinms' (int, optional): Window shift length in milliseconds. Defaults to 10 ms.
            - 'lp_order' (int, optional): Order of LP analysis. Defaults to 20.
        """
        if bool(extra_params) and 'random_coef_min' in extra_params:
            self.random_coef_min = self.extra_params['random_coef_min']
        else:
            self.random_coef_min = 0.5
        if bool(extra_params) and 'random_coef_max' in extra_params:
            self.random_coef_max = self.extra_params['random_coef_max']
        else:
            self.random_coef_max = 0.9       
        if bool(extra_params) and 'winLengthinms' in extra_params:
            self.winLengthinms = self.extra_params['winLengthinms']
        else:
            self.winLengthinms = 20
        if bool(extra_params) and 'shiftLengthinms' in extra_params:
            self.shiftLengthinms = self.extra_params['shiftLengthinms']
        else:
            self.shiftLengthinms = 10
        if bool(extra_params) and 'lp_order' in extra_params:
            self.lp_order = self.extra_params['lp_order']
        else:
            self.lp_order = 20

    def waveReadAsFloat(self, wavFileIn):
        """
        Read a waveform from a WAV file and convert it to floating-point format.

        Parameters:
        - wavFileIn (str): Path to the input WAV file.

        Returns:
        - sr (int): Sampling rate of the audio.
        - wavData (np.array): Waveform data in np.float32 format scaled between -1 and 1.
        """
        sr, wavdata = scipy.io.wavfile.read(wavFileIn)

        # Convert integer waveform to float between -1 and 1
        if wavdata.dtype is np.dtype(np.int16):
            wavdata = np.array(wavdata, dtype=np.float32) / np.power(2.0, 16 - 1)
        elif wavdata.dtype is np.dtype(np.int32):
            wavdata = np.array(wavdata, dtype=np.float32) / np.power(2.0, 32 - 1)
        elif wavdata.dtype is np.dtype(np.float32):
            pass
        else:
            print("Unknown waveform format %s" % (wavFileIn))
            sys.exit(1)
        return sr, wavdata

    def anonym(self, freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):
        """
        Anonymize audio using McAdams coefficient.

        Parameters:
        - freq (int): Sampling rate of the audio.
        - samples (np.array): Waveform data in np.float32 format scaled between -1 and 1.
        - winLengthinms (int, optional): Analysis window length in milliseconds. Defaults to 20 ms.
        - shiftLengthinms (int, optional): Window shift length in milliseconds. Defaults to 10 ms.
        - lp_order (int, optional): Order of LP analysis. Defaults to 20.
        - mcadams (float, optional): Alpha coefficients used in anonymization. Defaults to 0.8.

        Returns:
        - output_wav (np.array): Anonymized waveform data in np.int16 format.
        """
        eps = np.finfo(np.float32).eps
        samples = samples + eps

        # window length and shift (in sampling points)
        winlen = np.floor(winLengthinms * 0.001 * freq).astype(int)
        shift = np.floor(shiftLengthinms * 0.001 * freq).astype(int)
        length_sig = len(samples)

        # fft processing parameters
        NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
        # anaysis and synth window which satisfies the constraint
        wPR = np.hanning(winlen)
        K = np.sum(wPR) / shift
        win = np.sqrt(wPR / K)
        # number of of complete frames  
        Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int) 

        # Buffer for output signal
        # this is used for overlap - add FFT processing
        sig_rec = np.zeros([length_sig]) 

        # For each frame
        for m in np.arange(1, Nframes):

            # indices of the mth frame
            index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))    

            # windowed mth frame (other than rectangular window)
            frame = samples[index] * win 

            # get lpc coefficients
            a_lpc = librosa.core.lpc(y=frame + eps, order=lp_order)

            # get poles
            poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]

            #index of imaginary poles
            ind_imag = np.where(np.isreal(poles) == False)[0]

            #index of first imaginary poles
            ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]

            # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
            # values >1 expand the spectrum, while values <1 constract it for angles>1
            # values >1 constract the spectrum, while values <1 expand it for angles<1
            # the choice of this value is strongly linked to the number of lpc coefficients
            # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
            # a smaller lpc coefficients number allows for a bigger flexibility
            new_angles = np.angle(poles[ind_imag_con]) ** mcadams
            #new_angles = np.angle(poles[ind_imag_con])**path[m]

            # make sure new angles stay between 0 and pi
            new_angles[np.where(new_angles >= np.pi)] = np.pi
            new_angles[np.where(new_angles <= 0)] = 0  

            # copy of the original poles to be adjusted with the new angles
            new_poles = poles
            for k in np.arange(np.size(ind_imag_con)):
                # compute new poles with the same magnitued and new angles
                new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
                # applied also to the conjugate pole
                new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])            

            # recover new, modified lpc coefficients
            a_lpc_new = np.real(np.poly(new_poles))

            # get residual excitation for reconstruction
            res = scipy.signal.lfilter(a_lpc,np.array(1),frame)

            # reconstruct frames with new lpc coefficient
            frame_rec = scipy.signal.lfilter(np.array([1]),a_lpc_new,res)
            frame_rec = frame_rec * win    
            outindex = np.arange(m * shift, m * shift + len(frame_rec))

            # overlap add
            sig_rec[outindex] = sig_rec[outindex] + frame_rec

        sig_rec = (sig_rec / np.max(np.abs(sig_rec)) * (np.iinfo(np.int16).max - 1)).astype(np.int16)
        return sig_rec
        

    def anonymize(self, source_files, output_files):
        """
        Anonymize a batch of audio files.

        Parameters:
        - source_files (list): List of paths to the input audio files to be anonymized.
        - output_files (list): List of paths where the anonymized audio files will be saved.
        """
        # Loop through pairs of source and output files
        for line in tqdm(zip(source_files, output_files)):
            source_file, output_file = line

            # Read source audio file and convert to float
            sr, input_wav = self.waveReadAsFloat(source_file)

            # Generate a random coefficient in the range [0.5, 0.9]
            random_coef = random.uniform(self.random_coef_min, self.random_coef_max)

            # Anonymize the audio using the given parameters
            anonymized_wav = self.anonym(sr, input_wav, winLengthinms=self.winLengthinms, 
                                         shiftLengthinms=self.shiftLengthinms, lp_order=self.lp_order, 
                                         mcadams=random_coef)

            # Create the output directory if it doesn't exist
            pathlib.Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

            # Save the anonymized audio to the output file
            torchaudio.save(output_file, torch.tensor(anonymized_wav).unsqueeze(0), sr, 
                            encoding="PCM_S", bits_per_sample=16, format="wav")
