'''
    This script was written with the intention of parallelizing the feature extraction process since it takes a long time to run.

    Current Status: COMPLETE FAILURE🥲
'''

import os
import ray
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from scipy import fftpack
from utils import *

DATASET_PATH = '../Datasets/Accelerometer Data + TAC'
ACCELEROMETER_DATA = 'all_accelerometer_data_pids_13.csv'

sampling_rate = 40
long_seg_len, short_seg_len = 400, 40
TPB = 32

@ray.remote
class Worker(object):
    def __init__(self, short_segs, short_seg_fft_mag, long_seg_id, just_started, n_mfcc_features=13, short_seg_len=40):
        self.short_segs = short_segs
        self.short_seg_fft_mag = short_seg_fft_mag
        self.long_seg_id = long_seg_id
        self.just_started = just_started
        self.n_mfcc_features = n_mfcc_features
        self.short_seg_len = short_seg_len
        self.columns = []
        self.row = {}

    def apply_metric(self, metric_name, metric, seg_id, axis):
        if metric_name.split('_')[0] != 'fft':
            return metric(self.short_segs[seg_id][:, axis])
        if metric_name == 'fft_spectral_flux':
            fft_mag_prev = self.short_seg_fft_mag[seg_id - 1][:, axis] if seg_id > 0 else self.short_seg_fft_mag[seg_id][:, axis]
            return metric(self.short_seg_fft_mag[seg_id][:, axis], fft_mag_prev)
        return metric(self.short_seg_fft_mag[seg_id][:, axis])

    # Apply each metric for each short segment for each axis readings of the current long segment
    def compute_metric(self):
        for metric_name, metric in metrics.items():
            short_term_features = {'x': [], 'y': [], 'z': []}
            mfcc_coeffs = {'x': [], 'y': [], 'z': []}

            for short_seg_id in range(10):
                if metric_name != 'mfcc':
                    short_term_features['x'].append(self.apply_metric(metric_name, metric, short_seg_id, 0))
                    short_term_features['y'].append(self.apply_metric(metric_name, metric, short_seg_id, 1))
                    short_term_features['z'].append(self.apply_metric(metric_name, metric, short_seg_id, 2))
                else:
                # Compute the mfcc coefficients for the current short segment
                    mfcc_coeffs['x'].append(librosa.feature.mfcc(y=self.short_seg_fft_mag[short_seg_id][:, 0], sr=40, n_mfcc=self.n_mfcc_features, n_fft=self.short_seg_len // 2, n_mels=13))
                    mfcc_coeffs['y'].append(librosa.feature.mfcc(y=self.short_seg_fft_mag[short_seg_id][:, 1], sr=40, n_mfcc=self.n_mfcc_features, n_fft=self.short_seg_len // 2, n_mels=13))
                    mfcc_coeffs['z'].append(librosa.feature.mfcc(y=self.short_seg_fft_mag[short_seg_id][:, 2], sr=40, n_mfcc=self.n_mfcc_features, n_fft=self.short_seg_len // 2, n_mels=13))

            def mfcc_parallel(axis):
                ax1, ax2 = axis
                mfcc_coeffs[axis] = mfcc_coeffs[ax1] @ mfcc_coeffs[ax2].T
                short_term_features[axis] = mfcc_coeffs[axis][np.triu_indices(self.n_mfcc_features)]

            if metric_name == 'mfcc':
                mfcc_coeffs['x'] = np.array(mfcc_coeffs['x']).reshape(self.n_mfcc_features, 10)
                mfcc_coeffs['y'] = np.array(mfcc_coeffs['y']).reshape(self.n_mfcc_features, 10)
                mfcc_coeffs['z'] = np.array(mfcc_coeffs['z']).reshape(self.n_mfcc_features, 10)
                with ThreadPool(6) as pool:
                    pool.map(mfcc_parallel, ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'])

            # Now compute the summary statistics for the metric for the current long segment
            for axis, axis_data in short_term_features.items():
                if len(axis_data) == 0: continue
                for summary_stat_name, summary_stat in summary_stats.items():
                    self.row[f'{axis}_{metric_name}_{summary_stat_name}'] = summary_stat(axis_data)
                    if just_started:
                        self.columns.append(f'{axis}_{metric_name}_{summary_stat_name}')

        return self.row, columns

if __name__ == '__main__':
    columns = []
    just_started = True

    for acc_data in os.listdir(f'{DATASET_PATH}/clean_participant_data')[10:]:
        pid = acc_data.split('_')[0]
        data = pd.read_csv(f'{DATASET_PATH}/clean_participant_data/{acc_data}')
        feature_df = pd.DataFrame()
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
            row = {}

            pool = [Worker.remote(short_segs, short_seg_fft_mag, long_seg_id, just_started) for _ in range(os.cpu_count())]
            res = ray.get([actor.compute_metric.remote() for actor in pool])

            # Recover the order of the features
            # temp = []
            # for feat in columns:
            #     if not feat.endswith('_diff'): temp.append(row[feat])
            # row = temp
            # if just_started: columns.extend([f'{feature_name}_diff' for feature_name in columns])
            # just_started = False
            # if long_seg_id == 0:
            #     feature_df = pd.DataFrame(columns=columns)
            #     row *= 2
            # else:
            #     row.extend([row[i] - prev_row[i] for i in range(len(row))])
            # feature_df.loc[long_seg_id] = row
            # prev_row = temp
            start += short_seg_len // 2
            long_seg_id += 1

        # feature_df['ground_truth'] = ground_truth
        # feature_df.to_csv(f'{DATASET_PATH}/feature_data_new/{pid}.csv', index=False)
