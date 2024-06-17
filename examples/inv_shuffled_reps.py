fname_shuffled = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/20231202-ST001-cvn7009/GLMestimatesingletrialoutputs/extracted_voxs/lh_baseline200_20231202-ST001-cvn7009_extracted-voxs-parc_glasser_parcs-24_shuffled-reps.pkl'
fname = '/Users/gt/Library/CloudStorage/GoogleDrive-gretatu@mit.edu/My Drive/Research2020/Sentence7T/FMRI_FSAVERAGE/20231202-ST001-cvn7009/GLMestimatesingletrialoutputs/extracted_voxs/lh_baseline200_20231202-ST001-cvn7009_extracted-voxs-parc_glasser_parcs-24.pkl'

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d_shuffled = pickle.load(open(fname_shuffled, 'rb'))
d = pickle.load(open(fname, 'rb'))

data_shuffled = d_shuffled['betas_parc_3d'] # vertices, sentences, reps
data = d['betas_parc_3d']

(data_shuffled == data).all()

# For a given sentence and voxel, he same three trial values should exist in the shuffled and non-shuffled data
for sentence in range(data.shape[1]):
    for voxel in range(data.shape[0]):
        assert np.isin(data_shuffled[voxel, sentence, :], data[voxel, sentence, :]).all()

# plot scatter of rep 1 vs rep 1 for each sentence
for sentence in range(data.shape[1]):
    if sentence > 15:
        break
    plt.scatter(data[:, sentence, 0], data_shuffled[:, sentence, 0])
    plt.xlabel('Original')
    plt.ylabel('Shuffled')
    plt.title(f'Sentence {sentence}')
    plt.show()