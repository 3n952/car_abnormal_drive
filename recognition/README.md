# Recognize abnormal-driving using "You Only Watch Once (YOWO)"

## Visualize our work
(수정 필요)
<div align="center" style="width:image width px;">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/biking.gif" width=240 alt="방향지시등 불이행 차선변경">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/fencing.gif" width=240 alt="실선구간 차선변경">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/golf_swing.gif" width=240 alt="동시 차로변경">
</div>

<div align="center" style="width:image width px;"> 
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/catch.gif" width=240 alt="차선물기">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/brush_hair.gif" width=240 alt="2개 차로 연속 변경">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/pull_up.gif" width=240 alt="안전거리 미확보 차선변경">
</div>
<br/>
<br/>
  
# How to use YOWO for traffic dataset

## Installation
```bash
git clone https://github.com/3n952/car_abnormal_drive.git
cd recognition
```
### 1. Dataset
traffic dataset download : 'https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71566'

### 2. Download backbone pretrained weights

아래에서 각 Backbone에 대한 pretrained model을 다운 받을 수 있다.

* Darknet-19 weights
```bash
wget http://pjreddie.com/media/files/yolo.weights
```

* ResNeXt ve ResNet pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).

***NOTE:*** 기존 YOWO 학습 및 평가로 사용된 AVA, .. 등 dataset(action recognition)의 pretrained model은 traffic feature를 추출하기에 알맞지 않아 처음부터 train 시키는 것을 추천.

* For resource efficient 3D CNN architectures (ShuffleNet, ShuffleNetv2, MobileNet, MobileNetv2), pretrained models can be downloaded from [here](https://github.com/okankop/Efficient-3DCNNs).

### 3. Pretrained YOWO models

Pretrained models for traffic datasets can be downloaded from [업로드예정](https://www.dropbox.com).

## Running the code

* Training configuration(cfg)를 확인하고 조정할 것.
* 참고: [ava.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ava.yaml), [ucf24.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ucf24.yaml) and [jhmdb.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/jhmdb.yaml) files.
* training:
```bash
python main.py --cfg cfg/traffic.yaml
```

## Validating the model

* For traffic dataset, after each epoch, validation is performed and Video-mAP score is provided.
* For video_mAP, set the pretrained model in the correct yaml file and run:
```bash
python video_mAP.py --cfg cfg/traffic.yaml
```

## Running on a test video

* You can run traffic pretrained model on any test video with the following code:
```bash
python test_video_traffic.py --cfg cfg/traffic.yaml
```

### Citation
If you use this code or pre-trained models, please cite the following:

```bibtex
@InProceedings{kopuklu2019yowo,
title={You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization},
author={K{\"o}p{\"u}kl{\"u}, Okan and Wei, Xiangyu and Rigoll, Gerhard},
journal={arXiv preprint arXiv:1911.06644},
year={2019}
}
```