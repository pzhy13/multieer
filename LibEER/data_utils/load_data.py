import fnmatch
import json
import os

import scipy.io
from scipy.io import loadmat
import numpy as np
import multiprocessing as mp
from functools import partial
import mne
import xmltodict
import pickle
import mat73
from tqdm import tqdm
from scipy.signal import decimate

from data_utils.preprocess import preprocess, label_process
from .preprocess import lds


def get_data(setting=None):
    if setting is None:
        print(f"Error: Setting not set")

    # obtain data in the uniform formats, which load dataset and integrate into (session, subject, trail) format
    data, baseline, label, sample_rate, channels = get_uniform_data(setting.dataset, setting.dataset_path)
    # preprocess the eeg signal
    all_data, feature_dim = preprocess(data=data, baseline=baseline, sample_rate=sample_rate,
                                     pass_band=setting.pass_band, extract_bands=setting.extract_bands,
                                     sample_length=setting.sample_length, stride=setting.stride
                                     , time_window=setting.time_window, overlap=setting.overlap,
                                     only_seg=setting.only_seg if setting.dataset not in extract_dataset else True,
                                     feature_type=setting.feature_type,
                                     eog_clean=setting.eog_clean)

    all_data, all_label, num_classes = label_process(data=all_data, label=label, bounds=setting.bounds, onehot=setting.onehot, label_used=setting.label_used)
    return all_data, all_label, channels, feature_dim, num_classes


available_dataset = [
    "seed_raw", "seediv_raw", "deap", "deap_raw", "hci", "dreamer", "seed_de", "seed_de_lds", "seed_psd", "seed_psd_lds", "seed_dasm", "seed_dasm_lds"
    , "seed_rasm", "seed_rasm_lds", "seed_asm", "seed_asm_lds", "seed_dcau", "seed_dcau_lds", "seediv_de_lds", "seediv_de_movingAve",
    "seediv_psd_movingAve", "seediv_psd_lds", "faced_de", "faced_psd", "faced_de_lds", "faced_psd_lds", "seedv_raw", "seedv_de", "mped_raw","mped_feature",
    "mped_de_lds"
]

extract_dataset = {
    "seed_de", "seed_de_lds", "seed_psd", "seed_psd_lds", "seed_dasm", "seed_dasm_lds"
    , "seed_rasm", "seed_rasm_lds", "seed_asm", "see_und_asm_lds", "seed_dcau", "seed_dcau_lds", "seediv_de_lds", "seediv_de_movingAve",
    "seediv_psd_movingAve", "seediv_psd_lds", "faced_de", "faced_psd", "faced_de_lds", "faced_psd_lds","mped_feature","mped_de_lds"
}

def get_uniform_data(dataset, dataset_path):
    """
    Mainly aimed at the structure of different datasets,
    it is divided into the form of (session, subject, trail, channel, raw_data).
    :param dataset: the dataset used to train
    :param dataset_path: the dir of the dataset location
    :return: data, baseline, label, and sample rate of the original dataset
    """
    func = {
        "seed_raw": read_seed_raw,
        "deap": read_deap_preprocessed,
        "dreamer": read_dreamer,
        "deap_raw": read_deap_raw,
        "seediv_raw": read_seedIV_raw,
        "seedv_raw": read_seedV_raw,
        "mped_raw": read_mped_raw,
        "mped_feature": read_mped_feature,
        "mped_de_lds": read_mped_de_lds,
        "hci": read_hci
    }
    if dataset.startswith("seediv") and dataset != "seediv_raw":
        data, baseline, label, sample_rate, channels = read_seedIV_feature(dataset_path, feature_type=dataset[7:])
    elif dataset.startswith("seed") and not dataset.startswith("seediv") and not dataset.startswith("seedv") and dataset != "seed_raw":
        # call the read_seed_feature function when using the feature provided by seed official
        data, baseline, label, sample_rate, channels = read_seed_feature(dataset_path, feature_type=dataset[5:])
    elif dataset.startswith("faced") and dataset != "faced_raw":
        data, baseline, label, sample_rate, channels = read_faced_feature(dataset_path, feature_type=dataset[6:])
    else:
        data, baseline, label, sample_rate, channels = func[dataset](dataset_path)
    return data, baseline, label, sample_rate, channels

def read_faced_feature(dir_path, feature_type="de", label_type="emotion"):
    """
    input : 122 files(122 subjects)
    output : EEG signal with a trail as the basic unit
    output shape : (session(1), subject, trail, channel, raw_data), (session(1), subject, trail, label),
    Extract the EEG data of each subject from the faced dataset (feature)
    :param dir_path: The file location of the features in the faced dataset.
    :param feature_type: extract feature type
    :param label_type: Choose whether to use fine-grained("emotion"->9) or coarse-grained("valence"->3d) classification.
    :return: EEG features
    """
    is_lds = feature_type.endswith("_lds")
    dir_path += "/"+ feature_type[:-4].upper() if is_lds else feature_type.upper()
    # str(num).zfill(3)
    data = [[[] for _ in range(123)]]
    # emotion : Anger, Disgust, Fear, Sadness, Neutral, Amusement, Inspiration, Joy, Tenderness
    # valence : Negative, Neutral, Positive
    label = [[[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8] if label_type =="emotion" else [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2]  for _ in range(123)]]
    for i in range(123):
        file_path = dir_path + f"/sub{str(i).zfill(3)}.pkl.pkl"
        with open(file_path, "rb") as feature_file:
            sub_feature = pickle.load(feature_file)
            sub_feature = sub_feature.transpose(0,2,1,3)
            if is_lds:
                for j in range(len(sub_feature)):
                    sub_feature[j] = lds(sub_feature[j])
            data[0][i] = sub_feature
    return data, None, label, None, 32

def read_seed_raw(dir_path):
    # input : 45 files(3 sessions, 15 round) containing all 15 trails with a sampling rate of 200 Hz
    # output : EEG signal with a trail as the basic unit and sample rate of the original dataset
    # output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label)

    # Extract the EEG data of each subject from the SEED dataset, and partition the data of each session
    dir_path += "/Preprocessed_EEG"
    eeg_files = [['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
                  '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
                  '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
                  '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
                  '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
                 ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
                  '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
                  '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
                  '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
                  '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
                 ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
                  '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
                  '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
                  '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
                  '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
                 ]
    # Extract the label for all trail in three sessions
    label = np.array(loadmat(f"{dir_path}/label.mat")['label'])
    labels = np.tile(label[0]+1, (3, 15, 1))

    # create the empty list of (3, 15, 15) => (session, subject, trail)
    eeg_data = [[[[] for _ in range(15)] for _ in range(15)] for _ in range(3)]
    # Loop processing of EEG mat files
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            eeg_data[session_id] = pool.map(
                partial(parallel_read_seed_raw, dir_path), eeg_files[session_id])
    return eeg_data, None, labels, 200, 62

def parallel_read_seed_raw(dir_path, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    label_datas = []
    for i in range(15):
        trail_data = subject_data[keys[i]]
        trail_datas.append(trail_data[:,1:])
    return trail_datas

def read_seed_feature(dir_path, feature_type="de"):
    """
    input : 45 files(3 sessions, 15 round) containing all 15 trails with a sampling rate of 200 Hz
    output : EEG signal with a trail as the basic unit
    output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label),
    Extract the EEG data of each subject from the SEED dataset (feature) , and partition the data of each session
    :param dir_path: The file location of the features in the seed dataset.
    :param feature_type: extract feature type
    :return: the eeg features from seed dataset
    """

    dir_path += "/ExtractedFeatures"
    eeg_files = [['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
                  '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
                  '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
                  '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
                  '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
                 ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
                  '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
                  '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
                  '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
                  '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
                 ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
                  '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
                  '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
                  '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
                  '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
                 ]
    feature_index = {
        "de": 0, "de_lds": 1, "psd": 2, "psd_lds": 3, "dasm": 4, "dasm_lds": 5,
        "rasm": 6, "rasm_lds": 7, "asm": 8, "asm_lds": 9, "dcau": 10, "dcau_lds": 11
    }

    # Extract the label for all trail in three sessions, label shape : (15)
    label = np.array(loadmat(f"{dir_path}/label.mat")['label'])
    label = np.tile(label[0] + 1, (3, 15, 1))

    # Set index based on selected characteristics
    fi = feature_index[feature_type]

    eeg_data = [[] for _ in range(3)]
    # Define a function to read a single MAT file
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            result_session = pool.map(
                partial(parallel_read_seed_feature, fi, dir_path, label), eeg_files[session_id])
        for i in range(15):
            eeg_data[session_id].append(result_session[i])
    return eeg_data, None, label, None, 62

def parallel_read_seed_feature(fi, dir_path, label, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(15):
        trail_data = list(np.array(subject_data[keys[i * 12+fi]]).transpose((1, 0, 2)))
        trail_datas.append(trail_data)
    return trail_datas

def read_seedIV_raw(dir_path):
    # input : 45 files(3 sessions, 15 round)
    # output : EEG signal with a trail as the basic unit and sample rate of the original dataset
    # output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label)

    dir_path += "/eeg_raw_data"
    eeg_files = [['1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
                  '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
                  '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
                  '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
                  '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
                 ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
                  '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
                  '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
                  '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
                  '13_20151125.mat', '14_20151208.mat', '15_20150514.mat', ],
                 ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
                  '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
                  '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
                  '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
                  '13_20161130.mat', '14_20151215.mat', '15_20150527.mat', ]
                 ]

    # exctract the label for all trail in three sessions, label shape : (3, 24)
    label = np.zeros((3, 15, 24), dtype=int)
    ses_label1 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    ses_label2 = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
    ses_label3 = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    ses_label1s = np.tile(ses_label1, (1, 15, 1))
    ses_label2s = np.tile(ses_label2, (1, 15, 1))
    ses_label3s = np.tile(ses_label3, (1, 15, 1))
    label[0] = ses_label1s
    label[1] = ses_label2s
    label[2] = ses_label3s

    # Add a father session folder to each file
    for i, session in enumerate(eeg_files):
        eeg_files[i] = [f"{i + 1}/{sub_file}" for sub_file in session]

    # create the empty list of (3, 15, 24) => (session, subject, trail)
    eeg_data = [[[[] for _ in range(24)] for _ in range(15)] for _ in range(3)]
    # Loop processing of EEG mat files
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            eeg_data[session_id] = pool.map(
                partial(parallel_read_seedIV_raw, dir_path), eeg_files[session_id])
    return eeg_data, None, label, 200, 62

def parallel_read_seedIV_raw(dir_path, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(24):
        trail_data = subject_data[keys[i]]
        trail_datas.append(trail_data[:,1:])
    return trail_datas

def read_seedIV_feature(dir_path, feature_type="de_lds"):
    # 读取seed IV数据集
    # input file : three folder each contains one session of 15 subjects' eeg data
    # output shape : (session(3), subject, trail, channel, feature), (session(3), subject, trail, label)
    # use the feature under eeg_feature_smooth dir, it has 3 dir, each dir represent 15 subejct
    # in each dir, it contains 15 subject files
    dir_path += "/eeg_feature_smooth"
    eeg_files = [['1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
                  '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
                  '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
                  '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
                  '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
                 ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
                  '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
                  '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
                  '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
                  '13_20151125.mat', '14_20151208.mat', '15_20150514.mat',],
                 ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
                  '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
                  '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
                  '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
                  '13_20161130.mat', '14_20151215.mat', '15_20150527.mat', ]
                 ]

    #exctract the label for all trail in three sessions, label shape : (3, 24)
    label = np.zeros((3,15,24), dtype=int)
    ses_label1 = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    ses_label2 = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    ses_label3 = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    ses_label1s = np.tile(ses_label1, (1,15,1))
    ses_label2s = np.tile(ses_label2, (1,15,1))
    ses_label3s = np.tile(ses_label3, (1,15,1))
    label[0] = ses_label1s
    label[1] = ses_label2s
    label[2] = ses_label3s

    # Add a father session folder to each file
    for i, session in enumerate(eeg_files):
        eeg_files[i] = [f"{i+1}/{sub_file}" for sub_file in session]

    feature_index = {
        "de_movingAve": 0, "de_lds": 1, "psd_movingAve": 2, "psd_lds": 3
    }
    fi = feature_index[feature_type]

    eeg_data = [[] for _ in range(3)]
    # Define a function to read a single Mat file
    for ses_id, session_files in enumerate(eeg_files):
        with mp.Pool(processes=5) as pool:
            result_session = pool.map(
                partial(parallel_read_seedIV_feature, fi, dir_path, label), eeg_files[ses_id]
            )
        for i in range(15):
            eeg_data[ses_id].append(result_session[i])
    return eeg_data, None, label, None, 62

def parallel_read_seedIV_feature(fi, dir_path, label, file):
    subject_data = loadmat(f"{dir_path}/{file}")
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(24):
        trail_data = list(np.array(subject_data[keys[i*4+fi]].transpose((1,0,2))))
        trail_datas.append(trail_data)
    return trail_datas

def read_seedV_raw(dir_path):
    mne.set_log_level("ERROR")
    dir_path = dir_path + "EEG_raw/"
    eeg_files = [[
                  '1_1_20180804.cnt', '2_1_20180416.cnt', '3_1_20180414.cnt', '4_1_20180414.cnt',
                  '5_1_20180719.cnt', '6_1_20180713.cnt', '7_1_20180411.cnt', '8_1_20180717.cnt',
                  '9_1_20180724.cnt', '10_1_20180507.cnt', '11_1_20180510.cnt', '12_1_20180515.cnt',
                  '13_1_20180720.cnt', '14_1_20180420.cnt', '15_1_20180724.cnt', '16_1_20180805.cnt',],
                 [
                  '1_2_20180810.cnt', '2_2_20180419.cnt', '3_2_20180419.cnt', '4_2_20180417.cnt',
                  '5_2_20180728.cnt', '6_2_20180731.cnt', '7_2_20180418.cnt', '8_2_20180802.cnt',
                  '9_2_20180804.cnt', '10_2_20180524.cnt', '11_2_20180508.cnt', '12_2_20180508.cnt',
                  '13_2_20180806.cnt', '14_2_20180423.cnt', '15_2_20180807.cnt', '16_2_20180815.cnt',
                 ],
                 [
                     '1_3_20180808.cnt', '2_3_20180425.cnt', '3_3_20180424.cnt', '4_3_20180501.cnt',
                     '5_3_20180723.cnt', '6_3_20180802.cnt', '7_3_20180422.cnt', '8_3_20180726.cnt',
                     '9_3_20180728.cnt', '10_3_20180626.cnt', '11_3_20180522.cnt', '12_3_20180517.cnt',
                     '13_3_20180725.cnt', '14_3_20180427.cnt', '15_3_20180730.cnt', '16_3_20180813.cnt']
                 ]
    start_seconds = [[30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
                     [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
                     [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]]
    end_seconds = [[102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
                   [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
                   [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]]
    ses_id = 0
    data = [[],[],[]]
    label = np.zeros((3, 16, 15), dtype=int)
    ses_label1 = [4,1,3,2,0,4,1,3,2,0,4,1,3,2,0]
    ses_label2 = [2,1,3,0,4,4,0,3,2,1,3,4,1,2,0]
    ses_label3 = [2,1,3,0,4,4,0,3,2,1,3,4,1,2,0]
    ses_label1s = np.tile(ses_label1, (1, 16, 1))
    ses_label2s = np.tile(ses_label2, (1, 16, 1))
    ses_label3s = np.tile(ses_label3, (1, 16, 1))
    label[0] = ses_label1s
    label[1] = ses_label2s
    label[2] = ses_label3s
    total_files = sum(len(ses) for ses in eeg_files)
    pbar_total = tqdm(total=total_files, desc="Total Progress for seedV processing")

    for ses_id, ses_file in enumerate(eeg_files):
        with tqdm(total=len(ses_file), desc=f"Session {ses_id + 1}", leave=False) as pbar_session:
            for sub_file in ses_file:
                sub_data = []
                eeg_raw = mne.io.read_raw_cnt(dir_path + sub_file, verbose=False)
                useless_ch = ['M1', 'M2', 'VEO', 'HEO']
                eeg_raw.drop_channels(useless_ch)
                ch = eeg_raw.ch_names
                sub_raw_data = eeg_raw.get_data()
                for i in tqdm(range(15), desc="Processing trials", leave=False):
                    trail_data_i = sub_raw_data[:, start_seconds[ses_id][i] * 1000: end_seconds[ses_id][i] * 1000]
                    trail_downsampled_data_i = decimate(trail_data_i, q=5, ftype='fir', axis=1)
                    sub_data.append(trail_downsampled_data_i)
                data[ses_id].append(sub_data)
                pbar_session.update(1)
                pbar_total.update(1)
    pbar_total.close()
    return data, None, label, 200, 62

def read_mped_raw(dir_path):
    # available_set = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]
    # available_set = [2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    available_set  =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    # label = mat73.loadmat(dir_path + "label.mat")['label'].astype(int)[available_set] - 1
    dir_path = dir_path + "raw_EEG_signals/"
    # label = loadmat(dir_path + "raw_EEG_Label.mat")['eeg_label'][0][available_set] - 1

    label = np.array([1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 6, 3, 2, 1, 0, 5, 3, 4, 2, 6])
    print(label)
    print(len(label))
    # print(label[available_set])
    label = np.tile(label, (1, 23, 1))
    eeg_dirs = ['sunhaozhe', 'liaojiajie', 'taodan', 'wangshu', 'chuliuxi',
                 'taomengzhu', 'liyanni', 'chenmoran', 'caiyashi', 'yuanlin',
                 'chenyuanjie', 'panmengmeng', 'liuwei', 'zhangzhenhua', 'linmian',
                 'liaoshanshan', 'chengmengru', 'miuwenrong', 'weiruyuan', 'hetong',
                 'wangxiaobo', 'jianghanning', 'jialuhan']
    eeg_data = [[]]
    def extend_normal(sample):
        for i in range(len(sample)):
            features_min = np.min(sample[i])
            features_max = np.max(sample[i])
            sample[i] = (sample[i] - features_min) / (features_max - features_min)
        return sample
    for eeg_dir in eeg_dirs:
        sub_data = []
        eeg_file_d = dir_path + eeg_dir + "/" + eeg_dir + "_d.mat"
        eeg_file_dd = dir_path + eeg_dir + "/" + eeg_dir + "_dd.mat"
        data_d = loadmat(eeg_file_d) ## 1-17
        data_dd = loadmat(eeg_file_dd) ## 18-34
        for i in range(0,17):
            if i in available_set:
                key = "raw_eeg" + str(i+1)
                trail_data = data_d[key]
                # print(trail_data.shape)
                sub_data.append(decimate(trail_data[:62], q=5, ftype='fir', axis=1))
        for i in range(17,34):
            if i in available_set:
                key = "raw_eeg" + str(i+1)
                trail_data = data_dd[key]
                sub_data.append(decimate(trail_data[:62], q=5, ftype='fir', axis=1))
        sub_data = extend_normal(sub_data)
        eeg_data[0].append(sub_data)
    return eeg_data, None, label, 200, 62

def read_mped_feature(dir_path, feature_type="stft"):

    eeg_files = [
        "caiyashi.mat",     "chenmoran.mat",    "chuliuxi.mat",  "jialuhan.mat",      "liaojiajie.mat",    "linmian.mat",  "liyanni.mat",

        "panmengmeng.mat",  "taodan.mat",      "wangshu.mat",     "weiruyuan.mat",  "zhangzhenhua.mat",

        "chengmengru.mat",  "chenyuanjie.mat",  "hetong.mat",    "jianghanning.mat",  "liaoshanshan.mat",  "liuwei.mat",   "miuwenrong.mat",
        
        "sunhaozhe.mat",    "taomengzhu.mat",  "wangxiaobo.mat",  "yuanlin.mat"
    ]

    trial_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22, 24, 26, 20, 21, 23, 25, 27, 28, 29]

    label_list = [-1, 1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, -1, 0, 4, 5, 1, 1, 0, 6, 5, 3, 3, 2, 4, 2, 6]
    label = np.array([1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 6, 3, 2, 1, 0, 5, 3, 4, 2, 6])
    label = np.tile(label, (1, 23, 1))
    # for i in trial_list:/
        # print(str(label_list[i])+" ")
    # data = np.array([])  # 存储每一个subject的数组
    data = [[]]
    # label = [[]]
    # label = np.array([])

    def extend_normal(sample):
        for i in range(len(sample)):
            features_min = np.min(sample[i])
            features_max = np.max(sample[i])
            sample[i] = (sample[i] - features_min) / (features_max - features_min)
        return sample
    for subject in eeg_files:
        count = 0
        sub_data = []
        for i in trial_list:
            metaData = loadmat(dir_path + subject, verify_compressed_data_integrity=False)
            trMetaData = metaData['STFT'][0, i].astype('float')
            trMetaData = np.transpose(trMetaData, (1,0,2))  # (42,62,5)

            count += 1
            x = np.array(trMetaData)
            y = np.array([label_list[i], ] * x.shape[0])
            sub_data.append(extend_normal(x))

        data[0].append(sub_data)
    return data, None, label, None, 62

def read_mped_de_lds(dir_path):
    ## 已经处理好的de_lds特征
    data = pickle.load(dir_path)
    label = np.array([1, 0, 3, 2, 4, 5, 6, 0, 1, 2, 5, 6, 3, 4, 0, 4, 5, 1, 6, 3, 2, 1, 0, 5, 3, 4, 2, 6])
    label = np.tile(label, (1, 23, 1))
    return data, None, label, None, 62

def read_deap_preprocessed(dir_path):
    # 读取deap数据集
    # input file: 32 files contains 32 subject's eeg data
    # output shape : (session(1), subject, trail, channel, raw_data), (session(1), subject, trail, label)
    # under data_preprocess_python dir, it has 32.dat file, each represent one subject
    # every file contains two arrays:
    # data -> (trail(40), channel(40), data(8064))
    # label -> (trail(40), label(valence, arousal, dominance, liking))
    ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                'P4', 'P8', 'PO4', 'O2']
    data = [[]]
    label = [[]]
    fs = 128
    pre_time = 3
    end_time = 63
    pretrail = pre_time * fs

    eeg_files = ["s{}.dat".format(str(i).zfill(2)) for i in range(1,33)]
    for s_i, subject_file in enumerate(eeg_files):
        sub_data = pickle.load(open("{}/".format(dir_path)+subject_file, "rb"), encoding="latin")
        baseline = np.mean([sub_data['data'][:,:32,i*fs:(i+1)*fs] for i in range(3)], axis=0)
        for sec in range(pre_time, end_time):
            sub_data['data'][:, :32, sec*fs: (sec+1)*fs] -= baseline
        sub_data_list = []
        sub_label_list = []
        for t_i, (trail_data, trail_label) in enumerate(zip(sub_data['data'], sub_data['labels'])):
            # trail_data shape->(channels(32eeg, 8others), raw_data)
            # trail_label shape->(labels(valence, arousal, dominance, liking))
            sub_data_list.append(trail_data[:32,pretrail:])
            sub_label_list.append(trail_label)
        # sub_data_list -> (trail, channels, raw_data)
        # sub_label_list -> (trail, labels)
        data[0].append(sub_data_list)
        label[0].append(sub_label_list)
    # data -> (session(1), subject, trail, channel, raw_data)
    # label -> (session(1), subject, trail, channel, raw_data)
    return data, None, label, 128, 32

def read_deap_raw(dir_path):
    # 读取deap原始数据集
    # input file : 32 bdf files contains 32 subjects' eeg data
    # output shape : (session(1), subject, trail, channel, raw_data), (session(1), subject, trail, label)
    # under data_original dir, it has 32.bdf file, each represent one subject
    Geneva_ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                'P4', 'P8', 'PO4', 'O2']
    Twente_ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz',
                       'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4',
                       'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
    transfer_index =  [Twente_ch_names.index(s) for s in Geneva_ch_names]
    fs = 512
    pre_time = 3
    end_time = 63
    pretrail = pre_time * fs
    # when the code is 4, the experiment begin
    before_code = 3
    start_code1 = 4
    start_code2 = 1638148
    start_code3 = 5832452
    after_code = 5

    eeg_files = ["s{}.bdf".format(str(i).zfill(2)) for i in range(1,33)]
    label_file = ["s{}.dat".format(str(i).zfill(2)) for i in range(1,33)]
    all_raw_data = [[]]
    label = [[]]
    for s_i, subject_file in enumerate(eeg_files):
        sub_bdf_data = mne.io.read_raw_bdf("{}/data_original/".format(dir_path)+subject_file, preload=True
                                           , verbose=False)
        # print(sub_bdf_data.info['ch_names'])
        # get label easier
        label_data = pickle.load(open("{}/data_preprocessed_python/".format(dir_path)+label_file[s_i], "rb"),
                               encoding="latin")['labels']
        # read status code data
        status = np.array(sub_bdf_data.get_data()[47]).astype(int)
        changes = np.diff(status) != 0
        changes = np.insert(changes, 0, True)
        indices = np.where(changes)[0]
        # read raw data
        raw_data = np.array(sub_bdf_data.get_data()[:32])
        sub_raw_data = []
        sub_label = []
        pre_code = 0
        for begin, end in zip(indices, np.append(indices[1:], len(status))):
            # if s_i == 27:
            #     print(end-begin, status[begin])
            if pre_code == start_code1 or pre_code == start_code2 or pre_code == start_code3:
                # get last 60 seconds data points
                trail_raw_data = raw_data[:32, end-60*fs:end].tolist()
                if s_i < 22:
                    trail_raw_data = [trail_raw_data[tmp_i] for tmp_i in transfer_index]
                sub_raw_data.append(trail_raw_data)
            pre_code = status[begin]
        for t_i, trail_label in enumerate(label_data):
            sub_label.append(trail_label)
        all_raw_data[0].append(sub_raw_data)
        label[0].append(sub_label)
    return all_raw_data, None, label, 512, 32

def read_dreamer(dir_path, last_seconds = 60, base_seconds = 4):
    # input : 1 file (23 subjects' data)
    # subject data struct :
    #   Age, Gender, EEG, ECG, Valence(18 * 1), Arousal(18 * 1), Dominance(18 * 1)
    # subject's EEG data struct:
    #   sample rate : 128, num of electrodes : 14, num of subjects : 23
    #   electrodes : { 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'}
    # output shape : (session(1), subject, trail, channel, raw_data)
    file_path = dir_path + "/DREAMER.mat"
    data = loadmat(file_path)["DREAMER"]
    # data : [Data, EEG_sample_rate, ECG_sample_rate, EEG_electrodes, noOfSubjects, noOfVideoSequences
    # , Disclaimer, Provider, Version, Acknowledgement]
    # Data : [Age, Gender, EEG, ECG, ScoreValence, ScoreArousal, ScoreDominance]
    # EEG : [baseline, stimuli]
    # baseline & stimuli : [18, 1]
    #
    all_stimuli = [[[[] for _ in range(18)] for _ in range(23)]]
    all_base = [[[[] for _ in range(18)] for _ in range(23)]]
    all_labels = [[[[] for _ in range(18)] for _ in range(23)]]
    for subject in range(23):
        for trail in range(18):
            trail_stim = data[0,0]["Data"][0, subject]["EEG"][0, 0]["stimuli"][0, 0][trail, 0]
            trail_base = data[0,0]["Data"][0, subject]["EEG"][0, 0]["baseline"][0, 0][trail, 0]
            trail_valence = data[0,0]["Data"][0, subject]["ScoreValence"][0, 0][trail, 0]
            trail_arousal = data[0,0]["Data"][0, subject]["ScoreArousal"][0, 0][trail, 0]
            trail_dominance = data[0, 0]["Data"][0, subject]["ScoreDominance"][0, 0][trail, 0]
            trail_label = np.array([trail_valence, trail_arousal, trail_dominance])
            # print(trail_stim)
            # trail_stim shape : [128 * seconds(199), channel(14)]
            # trail_label shape : [3]
            all_stimuli[0][subject][trail] = trail_stim[-last_seconds*128:].transpose()
            all_base[0][subject][trail] = trail_base[-base_seconds*128:].transpose()
            all_labels[0][subject][trail] = trail_label
            # all_stimuli[0][subject][trail] shape : [channel(14), seconds(last_seconds) * sample rate(128)]
            # all_labels[0][subject][trail] shape : [3]
    return all_stimuli, all_base, all_labels, 128, 14

def read_hci(dir_path):
    # 30 subjects, [20, 20, 17, 20, 20, 20, 20, 20, 14, 20, 20, 0, 20, 20, 0, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    # input : 1 dir ( contains 1200 file )
    # output shape (session(1), subject, trail, channel, raw_data)
    baseline_sec = 30
    dir_path = dir_path + "/Sessions/"
    file_names = [name for name in os.listdir(dir_path)]
    emo_states = ['@feltVlnc', '@feltArsl']
    data = [[[] for _ in range(30)]]
    base = [[[] for _ in range(30)]]
    labels = [[[] for _ in range(30)]]

    for file in file_names:
        sub_dir = dir_path + file
        label_file = sub_dir + "/session.xml"
        with open(label_file) as f:
            label_info = xmltodict.parse('\n'.join(f.readlines()))
        label_info = json.loads(json.dumps(label_info))["session"]
        if not '@feltArsl' in label_info:
            continue
        trail_label = np.array([int(label_info[k]) for k in emo_states])
        sub = int(label_info['subject']['@id'])
        trail_file = [sub_dir+"/"+f for f in os.listdir(sub_dir) if fnmatch.fnmatch(f,'*.bdf')][0]
        raw = mne.io.read_raw_bdf(trail_file, preload=True, stim_channel='Status', verbose=False)
        events = mne.find_events(raw, stim_channel='Status', verbose=False)
        montage = mne.channels.make_standard_montage(kind='biosemi32')
        raw.set_montage(montage, on_missing='ignore')
        raw.pick(raw.ch_names[:32])
        start_samp, end_samp = events[0][0] + 1, events[1][0] - 1
        baseline = raw.copy().crop(raw.times[0], raw.times[end_samp])
        baseline = baseline.resample(128)
        baseline_data = baseline.to_data_frame().to_numpy()[:, 1:].swapaxes(1, 0)
        baseline_data = baseline_data[:, :baseline_sec * 128]
        baseline_data = baseline_data.reshape(32, baseline_sec, 128).mean(axis=1)

        trail_bdf = raw.copy().crop(raw.times[start_samp], raw.times[end_samp])
        trail_bdf = trail_bdf.resample(128)
        trail_data = trail_bdf.to_data_frame().to_numpy()[:,1:].swapaxes(1,0)
        data[0][sub-1].append(trail_data)
        base[0][sub-1].append(baseline_data)
        labels[0][sub-1].append(trail_label)

    filter_d_l_b = [(d,l,b) for d,l,b in zip(data[0], labels[0], base[0]) if l != []]
    data[0], labels[0], base[0] = zip(*filter_d_l_b) if filter_d_l_b else ([],[],[])
    return data, base, labels, 128, 32

