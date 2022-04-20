import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import fftpack, signal
from scipy.fftpack.realtransforms import dct
from scipy.stats import skew, kurtosis

DATASET_PATH = '../Datasets/Accelerometer Data + TAC'
ACCELEROMETER_DATA = 'all_accelerometer_data_pids_13.csv'

def zero_crossing_rate(data):
    return np.sum(np.abs(np.diff(np.sign(data)))) / (2 * (len(data) - 1))

def spectral_entropy(signal, n_short_blocks=10, eps=1e-9):
    """Computes the spectral entropy"""
    # number of frame samples
    num_frames = len(signal)
    # total spectral energy
    total_energy = np.sum(signal ** 2)
    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]
    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()
    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)
    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy

def spectral_centroid(fft_magnitude, sampling_rate=40, eps=1e-9):
    ind = (np.arange(1, len(fft_magnitude) + 1)) * (sampling_rate / (2.0 * len(fft_magnitude)))
    Xt = fft_magnitude.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps
    # Centroid:
    centroid = (NUM / DEN)
    # Normalize:
    centroid = centroid / (sampling_rate / 2.0)
    return centroid

def spectral_spread(fft_magnitude, sampling_rate=40, eps=1e-9):
    ind = (np.arange(1, len(fft_magnitude) + 1)) * (sampling_rate / (2.0 * len(fft_magnitude)))
    Xt = fft_magnitude.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps
    # Spread:
    spread = np.sqrt(np.sum(((ind - (NUM / DEN)) ** 2) * Xt) / DEN)
    # Normalize:
    spread = spread / (sampling_rate / 2.0)
    return spread

def spectral_flux(fft_magnitude, previous_fft_magnitude, eps=1e-9):
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum((fft_magnitude / fft_sum - previous_fft_magnitude / previous_fft_sum) ** 2)
    return sp_flux

def spectral_rolloff(signal, c=0.90, eps=1e-9):
    energy = np.sum(signal ** 2)
    fft_length = len(signal)
    threshold = c * energy
    # Find the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    cumulative_sum = np.cumsum(signal ** 2) + eps
    a = np.nonzero(cumulative_sum > threshold)[0]
    sp_rolloff = 0.0
    if len(a) > 0: sp_rolloff = np.float64(a[0]) / (float(fft_length))
    return sp_rolloff

def spectral_peak_ratio(fft_magnitude, eps=1e-9):
    # Ratio of largest peak to second largest peak
    peaks = sorted(fft_magnitude, reverse=True)
    if len(peaks) < 2: return 0.0
    return peaks[0] / (peaks[1] + eps)

def max_frequency(fft_magnitude, sampling_rate=40):
    max_freq = np.argmax(fft_magnitude)
    max_freq *= (sampling_rate / (2.0 * len(fft_magnitude)))
    return max_freq

def mfcc_filter_banks(sampling_rate, num_fft, lowfreq=133.33, linc=200 / 3,
                      logsc=1.0711703, num_lin_filt=13, num_log_filt=27):
    """
    Computes the triangular filterbank for MFCC computation 
    (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    if sampling_rate < 8000:
        nlogfil = 5

    # Total number of filters
    num_filt_total = num_lin_filt + num_log_filt

    # Compute frequency points of the triangle:
    frequencies = np.zeros(num_filt_total + 2)
    frequencies[:num_lin_filt] = lowfreq + np.arange(num_lin_filt) * linc
    frequencies[num_lin_filt:] = frequencies[num_lin_filt - 1] * logsc ** \
                                 np.arange(1, num_log_filt + 3)
    heights = 2. / (frequencies[2:] - frequencies[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((num_filt_total, num_fft))
    nfreqs = np.arange(num_fft) / (1. * num_fft) * sampling_rate

    for i in range(num_filt_total):
        low_freqs = frequencies[i]
        cent_freqs = frequencies[i + 1]
        high_freqs = frequencies[i + 2]

        lid = np.arange(np.floor(low_freqs * num_fft / sampling_rate) + 1,
                        np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        dtype=np.int32)
        lslope = heights[i] / (cent_freqs - low_freqs)
        rid = np.arange(np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        np.floor(high_freqs * num_fft / sampling_rate) + 1,
                        dtype=np.int32)
        rslope = heights[i] / (high_freqs - cent_freqs)
        print(lid)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_freqs)
        fbank[i][rid] = rslope * (high_freqs - nfreqs[rid])

    return fbank, frequencies

def mfcc(fft_magnitude, fbank, num_mfcc_feats, eps=1e-9):
    """
    Computes the MFCCs of a frame, given the fft mag
    ARGUMENTS:
        fft_magnitude:  fft magnitude abs(FFT)
        fbank:          filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:           MFCCs (13 element vector)
    Note:    MFCC calculation is, in general, taken from the 
             scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more 
         compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = np.log10(np.dot(fft_magnitude, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:num_mfcc_feats]
    return ceps

def avg_power(sig):
    _, power = signal.welch(sig, 40, nperseg=len(sig))
    return np.mean(power)

def rms(signal):
    return np.sqrt(np.mean(signal ** 2))

# if not os.path.exists(f'{DATASET_PATH}/feature_data_new'):
#     os.makedirs(f'{DATASET_PATH}/feature_data_new')
# else:
#     for file in os.listdir(f'{DATASET_PATH}/feature_data_new'):
#         os.remove(f'{DATASET_PATH}/feature_data_new/{file}')

metrics = {
    'mean': np.mean, 
    'std': np.std,
    'avg_abs_dev': lambda x: np.mean(np.abs(x - np.mean(x))),
    'min_raw': np.min,
    'max_raw': np.max,
    'min_abs': lambda x: np.min(np.abs(x)),
    'max_abs': lambda x: np.max(np.abs(x)), 
    'median': np.median, 
    'inter_quartile_range': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
    'zero_crossing_rate': zero_crossing_rate, 
    'skewness': skew, 
    'kurtosis': kurtosis, 
    'spectral_entropy': spectral_entropy, 
    'fft_spectral_entropy': spectral_entropy,
    'fft_spectral_centroid': spectral_centroid, 
    'fft_spectral_spread': spectral_spread,
    'fft_spectral_rolloff': spectral_rolloff,
    'fft_spectral_peak_ratio': spectral_peak_ratio,
    'avg_power': avg_power,
    'rms': rms,
    'max_freq': max_frequency,
    'fft_spectral_flux': spectral_flux,
    'mfcc': None # Mel Frequency Cepstral Coefficients (MFCC)
}

n_mfcc_features = 13
sampling_rate = 40
long_seg_len, short_seg_len = 400, 40

summary_stats = {
    'mean': np.mean,
    'variance': np.var,
    'min': np.min,
    'max': np.max,
    'lower_third_mean': lambda x: np.mean(sorted(x)[:len(x) // 3]),
    'upper_third_mean': lambda x: np.mean(sorted(x)[len(x) // 3:])
}

for acc_data in os.listdir(f'{DATASET_PATH}/clean_participant_data')[10:]:
    pid = acc_data.split('_')[0]
    data = pd.read_csv(f'{DATASET_PATH}/clean_participant_data/{acc_data}')
    feature_df = pd.DataFrame()
    columns = []
    ground_truth = []
    start, long_seg_id = 0, 0

    # Assume uniform frequency over the data samples
    # A long segment is of 10 seconds (400 samples) and a short segment is of 1 second (40 samples)
    # Long windows are segmented with 50% overlap
    # Short segments are sub-segments of the long segments in order to calculate features from a two-tiered approach
    for end in tqdm(range(long_seg_len, len(data), short_seg_len // 2), desc=f'Processing {pid}'):
        # Split the current long segment into 10 short segments
        short_segs = np.array_split(data.loc[start:end - 1, ['x', 'y', 'z']].to_numpy(), long_seg_len // short_seg_len)
        # Compute the fft magnitude spectrum for each short segment
        short_seg_fft_mag = [np.apply_along_axis(fftpack.fft, 1, short_seg) for short_seg in short_segs]
        short_seg_fft_mag = [np.abs(fft_mag) for fft_mag in short_seg_fft_mag]
        # Find the ground truth label for the current long segment (the most frequent label in the segment)
        ground_truth.append(data.loc[start:end - 1, 'ground_truth'].value_counts().idxmax())
        # Store the feature values for the current long segment
        row = []

        # Apply each metric for each short segment for each axis readings of the current long segment
        for metric_name, metric in metrics.items():
            short_term_features = {'x': [], 'y': [], 'z': []}
            mfcc_coeffs = {'x': [], 'y': [], 'z': []}

            def apply_metric(seg_id, axis):
                if metric_name.split('_')[0] != 'fft':
                    return metric(short_segs[seg_id][:, axis])
                if metric_name == 'fft_spectral_flux':
                    fft_mag_prev = short_seg_fft_mag[seg_id - 1][:, axis] if seg_id > 0 else short_seg_fft_mag[seg_id][:, axis]
                    return metric(short_seg_fft_mag[seg_id][:, axis], fft_mag_prev)
                return metric(short_seg_fft_mag[seg_id][:, axis])

            for short_seg_id in range(10):
                if metric_name != 'mfcc':
                    short_term_features['x'].append(apply_metric(short_seg_id, 0))
                    short_term_features['y'].append(apply_metric(short_seg_id, 1))
                    short_term_features['z'].append(apply_metric(short_seg_id, 2))
                else:
                # Compute the mfcc coefficients for the current short segment
                    mfcc_coeffs['x'].append(librosa.feature.mfcc(y=short_seg_fft_mag[short_seg_id][:, 0], sr=40, n_mfcc=n_mfcc_features, n_fft=short_seg_len // 2, n_mels=13))
                    mfcc_coeffs['y'].append(librosa.feature.mfcc(y=short_seg_fft_mag[short_seg_id][:, 1], sr=40, n_mfcc=n_mfcc_features, n_fft=short_seg_len // 2, n_mels=13))
                    mfcc_coeffs['z'].append(librosa.feature.mfcc(y=short_seg_fft_mag[short_seg_id][:, 2], sr=40, n_mfcc=n_mfcc_features, n_fft=short_seg_len // 2, n_mels=13))

            if metric_name == 'mfcc':
                mfcc_coeffs['x'] = np.array(mfcc_coeffs['x']).reshape(n_mfcc_features, 10)
                mfcc_coeffs['y'] = np.array(mfcc_coeffs['y']).reshape(n_mfcc_features, 10)
                mfcc_coeffs['z'] = np.array(mfcc_coeffs['z']).reshape(n_mfcc_features, 10)
                mfcc_coeffs['xx'] = mfcc_coeffs['x'] @ mfcc_coeffs['x'].T
                mfcc_coeffs['yy'] = mfcc_coeffs['y'] @ mfcc_coeffs['y'].T
                mfcc_coeffs['zz'] = mfcc_coeffs['z'] @ mfcc_coeffs['z'].T
                mfcc_coeffs['xy'] = mfcc_coeffs['x'] @ mfcc_coeffs['y'].T
                mfcc_coeffs['xz'] = mfcc_coeffs['x'] @ mfcc_coeffs['z'].T
                mfcc_coeffs['yz'] = mfcc_coeffs['y'] @ mfcc_coeffs['z'].T
                short_term_features['xx'] = mfcc_coeffs['xx'][np.triu_indices(n_mfcc_features)]
                short_term_features['yy'] = mfcc_coeffs['yy'][np.triu_indices(n_mfcc_features)]
                short_term_features['zz'] = mfcc_coeffs['zz'][np.triu_indices(n_mfcc_features)]
                short_term_features['xy'] = mfcc_coeffs['xy'][np.triu_indices(n_mfcc_features)]
                short_term_features['xz'] = mfcc_coeffs['xz'][np.triu_indices(n_mfcc_features)]
                short_term_features['yz'] = mfcc_coeffs['yz'][np.triu_indices(n_mfcc_features)]

            # Now compute the summary statistics for the metric for the current long segment
            for axis, axis_data in short_term_features.items():
                if len(axis_data) == 0: continue
                for summary_stat_name, summary_stat in summary_stats.items():
                    row.append(summary_stat(axis_data))
                    if long_seg_id == 0:
                        columns.append(f'{axis}_{metric_name}_{summary_stat_name}')

        temp = row
        if long_seg_id == 0:
            columns.extend([f'{feature_name}_diff' for feature_name in columns])
            feature_df = pd.DataFrame(columns=columns)
            row *= 2
        else:
            row.extend([row[i] - prev_row[i] for i in range(len(row))])
        feature_df.loc[long_seg_id] = row
        prev_row = temp
        start += short_seg_len // 2
        long_seg_id += 1

    feature_df['ground_truth'] = ground_truth
    feature_df.to_csv(f'{DATASET_PATH}/feature_data_new/{pid}.csv', index=False)
