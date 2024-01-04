from glob import glob
import json
from tools.data_sampling import preprocess_disease, divided_algorithm, print_statistics, copy_files_with_parents
import math
import pandas as pd
from collections import deque
import os
import pydicom
from tqdm import tqdm
from collections import Counter

if __name__ == '__main__':
    # Directory setting
    DATA_ROOT = '/workspace/data/bbox/ori/1.Datasets'
    DES_ROOT = '/workspace/data/bbox/subset'
    clinical_Info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/CLINICALINFO/*.json', recursive=True)
    # data_ratios = [.6, .2, .2]
    data_ratios = [.2, .2, .6]
    # subsets = ['train', 'val', 'test']
    subsets = ['val', 'test', 'train']

    std_dict = {
        'Ankle Arthritis': 'COR',
        'OLT': 'COR',
        'Tarsal Coalition': 'COR',
        'Accessory Navicular': 'AXL',
        'Flatfoot or Cavus': 'SAG',
    }
    infos = []
    for path in clinical_Info_paths:
        with open(path, 'r') as fp:
            db = json.load(fp)
        # assert len(db['Disease']) == 1
        disease = db['Disease'][0]
        disease_detail = db['DiseaseDetail'][0] if len(db['DiseaseDetail']) == 1 else ''
        disease_detail = preprocess_disease(disease_detail)
        if disease_detail not in std_dict.keys():
            continue
        _info = {
            'case_id': db['Case_ID'],
            'disease': disease,
            'sex': db['Sex'],
            'Age': db['Age'],
            'disease_detail': disease_detail,
            'generation': db['Age'] // 10
        }
        infos.append(_info)

    df = pd.DataFrame.from_records(infos)
    print('The number of entire patients: ', len(df.index))
    print('Sampling ratios: ', data_ratios)

    df_for_divided = df.copy()
    df_subset = {}
    std_ratio = 1.
    for i, (subset, ratio) in enumerate(zip(subsets, data_ratios)):
        # Data
        N_p = len(df_for_divided.index)
        if i == 0:
            N = math.ceil(N_p * ratio)
            dratio = ratio
        else:
            N = math.ceil(N_p * ratio / std_ratio)
            dratio += ratio

        df_divided, _ = divided_algorithm(df_for_divided, deque(['disease_detail', 'generation', 'sex']), N, ratio=dratio)
        # calculate left dataframe
        for _, row in df_divided.iterrows():
            df_for_divided.drop(index=df_for_divided[df_for_divided['case_id'] == row['case_id']].index, inplace=True)
        print(f'{subset}: {len(df_divided.index)}')
        df_subset[subset] = df_divided
        std_ratio -= ratio

    for subset in subsets:
        print(f'--------{subset}--------')
        print_statistics(df_subset[subset])
        print()
    for subset, df_subset in df_subset.items():
        desc_path = f'{DES_ROOT}/{subset}/1.Datasets'
        for idx, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0]):
            tag = std_dict[row['disease_detail']]
            _info = {'case_id': row["case_id"]}
            # bbox
            bbox_label_paths = [x for x in glob(f'{DATA_ROOT}/2.라벨링데이터/**/{row["case_id"]}/**/{tag}/*.json', recursive=True)]
            for lpath in bbox_label_paths:
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

            clinical_info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/{row["case_id"]}/CLINICALINFO/*.json', recursive=True)
            for info_path in clinical_info_paths:
                copy_files_with_parents(info_path, os.path.dirname(info_path.replace(DATA_ROOT, '')), desc_path)
        print('End split')
