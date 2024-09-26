# recognize abnormal driving using "You Only Watch Once (YOWO)"

# YOWO 
PyTorch implementation of the article "You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization". The repositry contains code for real-time spatiotemporal action localization with PyTorch.
Presenting ***YOWO*** (***Y**ou **O**nly **W**atch **O**nce*), a unified CNN architecture for real-time spatiotemporal action localization in video stream. *YOWO* is a single-stage framework, the input is a clip consisting of several successive frames in a video, while the output predicts bounding box positions as well as corresponding class labels in current frame. Afterwards, with specific strategy, these detections can be linked together to generate *Action Tubes* in the whole video.

Since YOWO do not separate human detection and action classification procedures, the whole network can be optimized by a joint loss in an end-to-end framework. YOWO have carried out a series of comparative evaluations on two challenging representative datasets **UCF101-24** and **J-HMDB-21**. Our approach outperforms the other state-of-the-art results while retaining real-time capability, providing 34 frames-per-second on 16-frames input clips and 62 frames-per-second on 8-frames input clips.

## Visualize our work(traffic)
(수정 필요)
<br/>
<br/>
<div align="center" style="width:image width px;">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/ava3.gif" width=240 alt="ava_example_1">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/ava1.gif" width=240 alt="ava_example_2">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/ava4.gif" width=240 alt="ava_example_3">
</div>
<br/>
<br/>

<br/>
<div align="center" style="width:image width px;">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/biking.gif" width=240 alt="biking">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/fencing.gif" width=240 alt="fencing">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/golf_swing.gif" width=240 alt="golf-swing">
</div>

<div align="center" style="width:image width px;"> 
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/catch.gif" width=240 alt="catch">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/brush_hair.gif" width=240 alt="brush-hair">
  <img  src="https://github.com/wei-tim/YOWO/blob/master/examples/pull_up.gif" width=240 alt="pull-up">
</div>
<br/>
<br/>
  


# How to use YOWO for traffic dataset

## Installation
```bash
git clone https://github.com/3n952/car_abnormal_drive.git
cd YOWO
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

Pretrained models for traffic datasets can be downloaded from '수정'[here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0).

## Running the code

* Training configuration(cfg)를 확인하고 조정할 것.
* 참고: [ava.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ava.yaml), [ucf24.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/ucf24.yaml) and [jhmdb.yaml](https://github.com/wei-tim/YOWO/blob/master/cfg/jhmdb.yaml) files.
* training:
```bash
python main.py --cfg cfg/traffic.yaml
```

## Validating the model (수정필요)

* For AVA dataset, after each epoch, validation is performed and frame-mAP score is provided.

* For UCF101-24 and J-HMDB-21 datasets, after each validation, frame detections is recorded under 'jhmdb_detections' or 'ucf_detections'. From [here](https://www.dropbox.com/sh/16jv2kwzom1pmlt/AABL3cFWDfG5MuH9PwnjSJf0a?dl=0), 'groundtruths_jhmdb.zip' and 'groundtruths_jhmdb.zip' should be downloaded and extracted to "evaluation/Object-Detection-Metrics". Then, run the following command to calculate frame_mAP.

```bash
python evaluation/Object-Detection-Metrics/pascalvoc.py --gtfolder PATH-TO-GROUNDTRUTHS-FOLDER --detfolder PATH-TO-DETECTIONS-FOLDER

```

* For video_mAP, set the pretrained model in the correct yaml file and run:
```bash
python video_mAP.py --cfg cfg/ucf24.yaml
```

## Running on a text video

* You can run AVA pretrained model on any test video with the following code:
```bash
python test_video_ava.py --cfg cfg/ava.yaml
```

***UPDATEs:*** 
* YOWO is extended for AVA dataset. 
* Old repo is deprecated and moved to [YOWO_deprecated](https://github.com/wei-tim/YOWO/tree/yowo_deprecated) branch. 

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

### Acknowledgements
We thank [Hang Xiao](https://github.com/marvis) for releasing [pytorch_yolo2](https://github.com/marvis/pytorch-yolo2) codebase, which we build our work on top. 
