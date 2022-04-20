import numpy as np
from scipy.fftpack.realtransforms import dct
from scipy.stats import skew, kurtosis
from scipy import signal

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

summary_stats = {
    'mean': np.mean,
    'variance': np.var,
    'min': np.min,
    'max': np.max,
    'lower_third_mean': lambda x: np.mean(sorted(x)[:len(x) // 3]),
    'upper_third_mean': lambda x: np.mean(sorted(x)[len(x) // 3:])
}