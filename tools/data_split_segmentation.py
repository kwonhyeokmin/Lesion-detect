from glob import glob
import json

import numpy as np

from tools.data_sampling_segmentation import divided_algorithm, print_statistics, copy_files_with_parents
import math
import pandas as pd
from collections import deque
import os
import pydicom
from tqdm import tqdm
import argparse
import random
random.seed(0)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', required=True, help='Image type for sampling (AXL, SAG, COR)', choices=['AXL', 'SAG', 'COR'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # args
    args = make_args()

    # Directory setting
    DATA_ROOT = '/workspace/data/segmentation/ori/1.Datasets'
    DES_ROOT = '/workspace/data/segmentation/subset'
    tag = args.data_type
    json_paths = glob(f'{DATA_ROOT}/2.라벨링데이터/**/{tag}/*.json', recursive=True)
    data_ratios = [0.6, 0.2, 0.2]

    # Setting categories name
    cat_nm = ['1st Metatarsal', '2nd Metatarsal', '3rd Metatarsal', '4th Metatarsal', '5th Metatarsal',
              'Calcaneus', 'Fibula', 'MidFoot', 'Talus', 'Tibia']

    print('Load json files ...')
    infos = []
    for path in tqdm(json_paths, desc='load json'):
        with open(path, 'r') as fp:
            db = json.load(fp)
        if len(db['ArrayOfannotation_info']) < 1:
            print(path)
            continue

        # clinical info
        clinical_info = db['ClinicalInfo']
        case_id = clinical_info['Case_ID']
        age = clinical_info['Age']
        sex = clinical_info['Sex']
        for ann in db['ArrayOfannotation_info']:
            if ann['object_name'] != 'LabelPolyLine':
                continue
            _cat_nm = ann['preset_name']
            if _cat_nm not in cat_nm:
                continue
            _info = {
                'case_id': case_id,
                'path': path,
                'sex': sex,
                'Age': age,
                'generation': age // 10,
                'cat_nm': _cat_nm
            }
            infos.append(_info)

    df = pd.DataFrame.from_records(infos)
    print('The number of entire patients: ', len(df.index))
    print('Sampling ratios: ', data_ratios)
    df_for_divided = df.copy()
    df_subset = {}
    std_ratio = 1.

    # split by random
    label_paths = df['path'].unique()
    random.shuffle(label_paths)
    n_data_split = (np.array(data_ratios) * len(label_paths)).astype(int).tolist()
    split_data_paths = label_paths[:n_data_split[0]], \
        label_paths[n_data_split[0]:n_data_split[0]+n_data_split[1]], \
        label_paths[n_data_split[0]+n_data_split[1]:]
    for subset, paths in zip(['train', 'val', 'test'], split_data_paths):
        print(f'{subset}: {len(paths)}')
    for (subset, label_paths) in zip(['train', 'val', 'test'], split_data_paths):
        desc_path = f'{DES_ROOT}/{subset}/1.Datasets'
        for lpath in tqdm(label_paths, desc=subset):
            try:
                # check image path is valid
                img_path = lpath.replace('2.라벨링데이터', '1.원천데이터').replace('.json', '.dcm')
                dcm = pydicom.dcmread(img_path)
                dcm_img = dcm.pixel_array
                if dcm_img is None:
                    print(f'File not exist path is {img_path}')
                    continue
                # copy images and label files to destination directory
                copy_files_with_parents(img_path, os.path.dirname(img_path.replace(DATA_ROOT, '')), desc_path)
                copy_files_with_parents(lpath, os.path.dirname(lpath.replace(DATA_ROOT, '')), desc_path)
            except Exception as e:
                print(e)
                continue

    # for i, (subset, ratio) in enumerate(zip(['train', 'val', 'test'], data_ratios)):
    #     # Data
    #     N_p = len(df_for_divided.index)
    #     if i == 0:
    #         N = math.ceil(N_p * ratio)
    #     else:
    #         N = math.ceil(N_p * ratio / std_ratio)
    #
    #     df_divided, _ = divided_algorithm(df_for_divided, deque(['case_id', 'cat_nm', 'generation', 'sex']), N)
    #     # calculate left dataframe
    #     for _, row in df_divided.iterrows():
    #         df_for_divided.drop(
    #             index=df_for_divided[df_for_divided['case_id'] == row['case_id']].index,
    #             inplace=True)
    #     print(f'{subset}: {len(df_divided.index)}')
    #     df_subset[subset] = df_divided
    #     std_ratio -= ratio

    # for subset, df_subset in df_subset.items():
    #     desc_path = f'{DES_ROOT}/{subset}/1.Datasets'
    #     case_ids = df_subset['case_id'].unique()
    #
    #     for case_id in tqdm(case_ids):
    #         _info = {'case_id': case_id}
    #         # label path
    #         label_dir = f'{DATA_ROOT}/2.라벨링데이터/'
    #         label_paths = [x for x in glob(f'{DATA_ROOT}/2.라벨링데이터/**/{case_id}/**/{tag}/*.json', recursive=True)]
    #         for lpath in label_paths:
    #             # check image path is valid
    #             img_path = lpath.replace('2.라벨링데이터', '1.원천데이터').replace('.json', '.dcm')
    #             dcm = pydicom.dcmread(img_path)
    #             dcm_img = dcm.pixel_array
    #             if dcm_img is None:
    #                 print(f'File not exist path is {img_path}')
    #                 continue
    #             # copy images and label files to destination directory
    #             copy_files_with_parents(img_path, os.path.dirname(img_path.replace(DATA_ROOT, '')), desc_path)
    #             copy_files_with_parents(lpath, os.path.dirname(lpath.replace(DATA_ROOT, '')), desc_path)
    #
    #         clinical_info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/{case_id}/CLINICALINFO/*.json', recursive=True)
    #         for info_path in clinical_info_paths:
    #             copy_files_with_parents(info_path, os.path.dirname(info_path.replace(DATA_ROOT, '')), desc_path)
    #     print('End split')
