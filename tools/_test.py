from glob import glob
import os
from tools.data_sampling import copy_files_with_parents


# Directory setting
DATA_ROOT = '/workspace/data/sample/1.Datasets'
DES_ROOT = '/workspace/data/sample/1.Datasets'

dirs = glob(f'{DATA_ROOT}/2.라벨링데이터/**/AN*', recursive=True)
for _dir in dirs:
    clinical_Info_path = os.path.join(_dir.replace('2.라벨링데이터', '1.원천데이터'), 'CLINICALINFO')
    clinical_json_path = glob(f'{clinical_Info_path}/*.json')[0]
    print(clinical_json_path)
# clinical_Info_paths = glob(f'{DATA_ROOT}/1.원천데이터/**/CLINICALINFO/*.json', recursive=True)