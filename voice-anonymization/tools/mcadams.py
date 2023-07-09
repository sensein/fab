from tqdm import tqdm
import os
import pathlib
import torchaudio
import random
import numpy as np
import scipy.io.wavfile
import librosa
import torch


def f_read_raw_mat(filename, col, data_format='f4', end='l'):
    """data = f_read_raw_mat(filename, col, data_format='float', end='l')
    Read the binary data from filename
    Return data, which is a (N, col) array

    input
    -----
       filename: str, path to the binary data on the file system
       col:      int, number of column assumed by the data matrix
       format:   str, please use the Python protocal to write format
                 default: 'f4', float32
       end:      str, little endian 'l' or big endian 'b'?
                 default: 'l'
    output
    ------
       data: np.array, shape (N, col), where N is the number of rows
           decided by total_number_elements // col
    """
    f = open(filename, 'rb')
    if end == 'l':
        data_format = '<' + data_format
    elif end == 'b':
        data_format = '>' + data_format
    else:
        data_format = '=' + data_format
    datatype = np.dtype((data_format, (col,)))
    data = np.fromfile(f, dtype=datatype)
    f.close()
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:, 0]
    else:
        return data


def f_write_raw_mat(data, filename, data_format='f4', end='l'):
    """flag = write_raw_mat(data, filename, data_format='f4', end='l')
    Write data to file on the file system as binary data

    input
    -----
      data:     np.array, data to be saved
      filename: str, path of the file to save the data
      data_format:   str, data_format for numpy
                 default: 'f4', float32
      end: str   little endian 'l' or big endian 'b'?
                 default: 'l'

    output
    ------
      flag: bool, whether the writing is done or not
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input should be np.array")
        return False
    f = open(filename, 'wb')
    if len(data_format) > 0:
        if end == 'l':
            data_format = '<' + data_format
        elif end == 'b':
            data_format = '>' + data_format
        else:
            data_format = '=' + data_format
        datatype = np.dtype(data_format)
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f, '')
    f.close()
    return True


# Function to load waveform data
def waveReadAsFloat(wavFileIn):
    """ sr, wavData = wavReadToFloat(wavFileIn)
    Wrapper over scipy.io.wavfile
    Return:
        sr: sampling_rate
        wavData: waveform in np.float32 (-1, 1)
    """

    sr, wavdata = scipy.io.wavfile.read(wavFileIn)

    if wavdata.dtype is np.dtype(np.int16):
        wavdata = np.array(wavdata, dtype=np.float32) / \
                  np.power(2.0, 16 - 1)
    elif wavdata.dtype is np.dtype(np.int32):
        wavdata = np.array(wavdata, dtype=np.float32) / \
                  np.power(2.0, 32 - 1)
    elif wavdata.dtype is np.dtype(np.float32):
        pass
    else:
        print("Unknown waveform format %s" % (wavFileIn))
        sys.exit(1)
    return sr, wavdata


def anonym(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):
    """ output_wav = anonym(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8)

    Anonymization using McAdam coefficient.

    :input: freq, int, sampling rate in Hz, 16000 in this case
    :input: samples, np.array, (L, 1) where L is the length of the waveform
    :input: winLengthinms, int, analysis window length (ms), default 20 ms
    :input: shiftLengthinms, int, window shift (ms), default 10 ms
    :input: lp_order, int, order of LP analysis, default 20
    :input: mcadams, float, alpha coefficients, default 0.8

    :output: output_wav, np.array, same shape as samples
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

        # index of imaginary poles
        ind_imag = np.where(np.isreal(poles) == False)[0]

        # index of first imaginary poles
        ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]

        # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
        # values >1 expand the spectrum, while values <1 constract it for angles>1
        # values >1 constract the spectrum, while values <1 expand it for angles<1
        # the choice of this value is strongly linked to the number of lpc coefficients
        # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
        # a smaller lpc coefficients number allows for a bigger flexibility
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams
        # new_angles = np.angle(poles[ind_imag_con])**path[m]

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
        res = scipy.signal.lfilter(a_lpc, np.array(1), frame)

        # reconstruct frames with new lpc coefficient
        frame_rec = scipy.signal.lfilter(np.array([1]), a_lpc_new, res)
        frame_rec = frame_rec * win
        outindex = np.arange(m * shift, m * shift + len(frame_rec))

        # overlap add
        sig_rec[outindex] = sig_rec[outindex] + frame_rec

    sig_rec = (sig_rec / np.max(np.abs(sig_rec)) * (np.iinfo(np.int16).max - 1)).astype(np.int16)
    return sig_rec


def anonymize(source_files, output_files):
    for line in tqdm(zip(source_files, output_files)):
        source_file, output_file = line

        sr, input_wav = waveReadAsFloat(source_file)
        random_coef = random.uniform(0.5, 0.9)
        anonymized_wav = anonym(sr, input_wav, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=random_coef)

        pathlib.Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_file, torch.tensor(anonymized_wav).unsqueeze(0), sr, encoding="PCM_S",
                        bits_per_sample=16, format="wav")