# Lesion Detection

## install

각 컴퓨터 cuda 버전에 맞는 파이토치를 설치합니다.
[PYTORCH](https://pytorch.org/)
설치가 완료되면 아래 명령어를 통해 관련 라이브러리를 설치합니다.

```shell
pip install -r requirements.txt
```

## train
```shell
python3 train.py --workers 8 --device 0 --batch-size 32 --data data/grazpedwri.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name lesin-yolov7 --hyp data/hyp.scratch.p5.yaml
```