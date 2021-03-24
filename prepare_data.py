import glob
import os

data_path = 'epadb/test/data'
mfcc_path =  data_path + '/mfccs.ark'
ivectors_path =  data_path + '/ivectors.ark'

wav_scp_path = 'wav.scp'
spk2utt_path = 'spk2utt'

extract_features_using_kaldi(data_path, mfcc_path, ivectors_path, wav_scp_path, spk2utt_path)