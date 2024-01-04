from glob import glob
import json
from tools.data_sampling import preprocess_disease, divided_algorithm, print_statistics, copy_files_with_parents
from utils.datasets import filter_empty_img
import math
import pandas as pd
import random
import os
import pydicom
from tqdm import tqdm
import numpy as np
import cv2
from multiprocessing import Pool
from contextlib import contextmanager


def load_dcm(path):
    dcm = pydicom.dcmread(path)
    dcm_img = dcm.pixel_array
    dcm_img = dcm_img.astype(float)
    # Rescaling grey scale between 0-255
    dcm_img_scaled = (np.maximum(dcm_img, 0) / dcm_img.max()) * 255
    # Convert to uint
    dcm_img_scaled = np.uint8(dcm_img_scaled)

    img = cv2.cvtColor(dcm_img_scaled, cv2.COLOR_GRAY2BGR)
    return img


if __name__ == '__main__':
    # Directory setting
    DATA_ROOT = '/workspace/data/ori/1.Datasets'
    DES_ROOT = '/workspace/data/subset'
    clinical_Info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/CLINICALINFO/*.json', recursive=True)
    data_ratios = [.8, .1, .1]

    # get image list
    img_infos = []
    label_path_list = glob(f'{DATA_ROOT}/2.라벨링데이터/**/*.json', recursive=True)
    for lpath in tqdm(label_path_list, desc='load labels'):
        with open(lpath, 'r') as jp:
            db = json.load(jp)
            info_db = db['ClinicalInfo']
            arr_db = [x for x in db['ArrayOfannotation_info'] if x['object_name']=='LabelRect']
        if len(arr_db) < 1:
            continue
        case_id = info_db['Case_ID']
        dmc = info_db['DiseaseMajorCode']
        dd = preprocess_disease(info_db['DiseaseDetail'][0])
        img_path = lpath.replace('2.라벨링데이터', '1.원천데이터').replace('.json', '.dcm')
        frame = int(os.path.basename(img_path).replace('.dcm', '').split('_')[-1])
        tag = os.path.dirname(img_path).split('/')[-1]

        # bbox
        arr = arr_db[0]
        bbox_x1 = min(arr['start_pos']['X'], arr['end_pos']['X'])
        bbox_y1 = min(arr['start_pos']['Y'], arr['end_pos']['Y'])
        bbox_x2 = max(arr['end_pos']['X'], arr['start_pos']['X'])
        bbox_y2 = max(arr['end_pos']['Y'], arr['start_pos']['Y'])

        img_infos.append({
            'case_id': case_id,
            'dmc': dmc,
            'dd': dd,
            'img_path': img_path,
            'lpath': lpath,
            'frame': frame,
            'tag': tag,
            'bbox': [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
        })
    df = pd.DataFrame.from_records(img_infos, coerce_float=False)
    df['subset'] = None
    print(f'The number of Total dataset: {len(df)}')
    case_ids = df.case_id.unique()
    conditions = []
    results = []
    for case_id in tqdm(case_ids, desc='split data'):
        df_case = df[df['case_id'] == case_id]
        df_case_sorted = df_case.sort_values('frame')

        N = len(df_case_sorted)
        N_v = int(N * data_ratios[1])
        N_t = int(N * data_ratios[2])

        # filtering images
        filtered_frame = []
        weights = []
        for i, row in df_case_sorted.iterrows():
            img_path = row['img_path']
            tag = row['tag']
            frame = row['frame']
            dd = row['dd']
            img = load_dcm(img_path)
            h, w, _ = img.shape
            score = filter_empty_img(img, row['bbox'], pixel=20)
            if not score > .50:
                continue
            # if not filter_empty_img(img, [0, 0, w, h], thrs=0.50 if tag=='SAG' else 0.35):
            #     continue
            filtered_frame.append(frame)
            weights.append(score)
        if len(filtered_frame) < N_v+N_t:
            print(f'filtered_frame: {len(filtered_frame)}, (N_v+N_t): {N_v+N_t}, case_id: {case_id}, DiseaseDetail: {dd}')
            conditions.append((df['case_id'] == case_id))
            results.append('train')
            continue
        selected = random.choices(filtered_frame, weights=weights, k=N_v+N_t)
        v_v = random.choices(selected, k=N_v)
        f_selected = [x for x in selected if x not in v_v]
        v_t = random.choices(f_selected, k=N_t)
        conditions.extend([
            (df['case_id'] == case_id) & (df['frame'].isin(v_v)),
            (df['case_id'] == case_id) & (df['frame'].isin(v_t)),
            (df['case_id'] == case_id) & ~(df['frame'].isin(v_v) & df['frame'].isin(v_t))
        ])
        results.extend([
            'val', 'test', 'train'
        ])
    df['subset'] = np.select(conditions, results)
    # fill nan to train(default)
    # df = df.fillna('train')
    for subset in ['train', 'val', 'test']:
        print(f'The number of {subset} datasets: {len(df[df["subset"]==subset])}')

    def func_bbox_copy(bbox_label_paths, desc_path):
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

    def func_seg_copy(seg_label_paths, desc_path):
        for lpath in seg_label_paths:
            # copy label files to destination directory
            copy_files_with_parents(lpath, os.path.dirname(lpath.replace(DATA_ROOT, '')), desc_path)

    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    n_process = os.cpu_count() * 2
    pool = Pool(n_process)
    for i, row in tqdm(df.iterrows(), desc='copy files', total=df.shape[0]):
        subset = row['subset']
        tag = row['tag']
        desc_path = f'{DES_ROOT}/{subset}/1.Datasets'
        _info = {}
        # clinical info
        clinical_info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/{row["case_id"]}/CLINICALINFO/*.json', recursive=True)
        for info_path in clinical_info_paths:
            copy_files_with_parents(info_path, os.path.dirname(info_path.replace(DATA_ROOT, '')), desc_path)

        # bbox
        bbox_label_paths = [x for x in glob(f'{DATA_ROOT}/2.라벨링데이터/**/{row["case_id"]}/**/{tag}/*.json', recursive=True)]
        func_bbox_copy(bbox_label_paths, desc_path)
        # with poolcontext(processes=n_process) as pool:
        #     pool.map(func_bbox_copy, zip(bbox_label_paths))
        # pool.map(func_bbox_copy, zip(bbox_label_paths, desc_path))
        # segmentation
        seg_label_paths = [x for x in glob(f'{DATA_ROOT}/2.라벨링데이터/**/{row["case_id"]}/**/{tag}/*.json', recursive=True)]
        func_seg_copy(seg_label_paths, desc_path)
        # with poolcontext(processes=n_process) as pool:
        #     pool.map(func_seg_copy, zip(seg_label_paths))

        _info.update({
            f'BBOX-{tag}': len(bbox_label_paths),
            f'SEG-{tag}': len(seg_label_paths)
        })
    print('End split')
    print('process finished')
