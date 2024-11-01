# Classify car abnormal driving using Video swin transformer

## Visualize our work
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

# How to use

## Installation
```bash
git clone https://github.com/3n952/car_abnormal_drive.git
cd classification
```
### 1. Dataset
traffic dataset download : 'https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71566'

### 2. Pretrained Video swin transformer models

Pretrained models(video swin transformer) for traffic datasets can be downloaded from [업로드예정](https://www.dropbox.com).

## 3. Running the code

* training:
```bash
python train.py
```

## Validating the model

* For traffic dataset, after each epoch, validation is performed and Video-mAP score is provided.

```bash
python test.py
```