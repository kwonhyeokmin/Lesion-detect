import json
import os
import random
import shutil
from glob import glob
import pandas as pd
import math
from collections import deque
from tqdm import tqdm
import pydicom


def preprocess_disease(name):
    _cls_name = name.replace('-C:', '').replace(';', '')
    try:
        split_std_idx = _cls_name.index(')')
        major_cls_name = _cls_name[:split_std_idx + 1].strip()
        minor_cls_name = _cls_name[split_std_idx + 1:].strip()

        # cls_name = f'{major_cls_name} {minor_cls_name}'.strip()
        subtitle_idx = major_cls_name.index('(')
        cls_name = major_cls_name[:subtitle_idx].strip()
    except ValueError:
        cls_name = _cls_name
    return cls_name


def print_statistics(df, cols=None):
    # *******************
    # Print Statistics
    # *******************
    if cols is None:
        cols = ['disease', 'sex', 'disease_detail', 'generation']
    print('Dataframe Sample')
    print('--------------------')
    print(df.head(5))
    print()

    for col in cols:
        # statistic of disease
        print(f'Statistics of {col}')
        print('--------------------')
        print(df.groupby(col)[col].count())
        print()


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def copy_files_with_parents(file_path, base_dir, desc):
    try:
        target_d = f'{desc}{base_dir}'
        make_folder(target_d)
        shutil.copy2(file_path, target_d)
    except Exception as e:
        print(e)
        return False
    return True


def divided_algorithm(df, queue, N, ratio=1):
    _df = df.copy()
    # Calculate the number of images by case_id
    std_col = queue.popleft()
    C = set(_df[std_col])

    divided = pd.DataFrame(columns=_df.columns)
    N_count = _df.groupby(std_col)[std_col].count()
    while len(C) > 0 and N > 0:
        N_std = N_count.min()
        C_std = N_count.idxmin()

        if N_std < N / len(C):
            NC = math.ceil(N_std * ratio)
            N = N - NC
            C.remove(C_std)
            sample_c = _df[_df[std_col] == C_std].sample(n=NC, random_state=1)
            divided = pd.concat([divided, sample_c], ignore_index=True)
            _df.drop(sample_c.index, inplace=True)
            N_count = N_count.drop(labels=C_std)
        else:
            # recursive
            if len(queue) > 0:
                result, N = divided_algorithm(_df, queue, N)
                divided = pd.concat([divided, result], ignore_index=True)
            else:
                # choice by random
                for c in C:
                    sample_c = _df[_df[std_col]==c].sample(n=math.ceil(N / len(C)), random_state=1)
                    divided = pd.concat([divided, sample_c], ignore_index=True)
                return divided, 0
    return divided, N


if __name__ == '__main__':
    std_dict = {
        'Ankle Arthritis': 'COR',
        'OLT': 'COR',
        'Tarsal Coalition': 'COR',
        'Accessory Navicular': 'AXL',
        'Flatfoot or Cavus': 'SAG',
    }
    # Directory setting
    DATA_ROOT = '/workspace/data/bbox/1.Datasets'
    DES_ROOT = '/workspace/data/bbox/ori/1.Datasets'
    clinical_Info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/CLINICALINFO/*.json', recursive=True)

    infos = []
    for path in clinical_Info_paths:
        with open(path, 'r') as fp:
            db = json.load(fp)
        if len(db['Disease']) < 1:
            print(path)
            continue
        disease = db['Disease'][0]
        disease_detail = db['DiseaseDetail'][0] if len(db['DiseaseDetail']) == 1 else ''
        disease_detail = preprocess_disease(disease_detail)
        if disease_detail not in std_dict.keys():
            continue
        _info = {
            'case_id': db['Case_ID'],
            'path': path,
            'disease': disease,
            'sex': db['Sex'],
            'Age': db['Age'],
            'disease_detail': disease_detail,
            'generation': db['Age'] // 10
        }
        infos.append(_info)

    df = pd.DataFrame.from_records(infos)
    df_n = df[df['disease']=='N']
    df_p = df[df['disease']!='N']
    print('The number of sampled data: ', len(df_p.index))

    json_info = []
    for idx, row in tqdm(df_p.iterrows(), total=df_p.shape[0]):
        _info = {'case_id': row['case_id']}
        tag = std_dict[row['disease_detail']]
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
            copy_files_with_parents(img_path, os.path.dirname(img_path.replace(DATA_ROOT, '')), DES_ROOT)
            copy_files_with_parents(lpath, os.path.dirname(lpath.replace(DATA_ROOT, '')), DES_ROOT)

        _info.update({
            f'BBOX-{tag}': len(bbox_label_paths),
        })
        # clinical info
        copy_files_with_parents(row['path'], os.path.dirname(row['path'].replace(DATA_ROOT, '')), DES_ROOT)
        json_info.append(_info)
    print('End sampling')
    # summary save
    patient_divided = pd.merge(df_p, pd.DataFrame.from_records(json_info))

    print(patient_divided.groupby('disease_detail')
          [['BBOX-SAG', 'BBOX-AXL', 'BBOX-COR']].sum())
