# Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians

### 환경 구성
: [Gaussian-Head-Avatar](https://github.com/YuelangX/Gaussian-Head-Avatar)구동을 위한 환경을 구축하고 명시합니다.

<br>

|구성|사양|비고|
|:--:|:--|:--|
|OS|Ubuntu 20.04||
|CPU|AMD EPYC 7V13 64-Core Processor, core(24)||
|GPU|NVIDIA A100||
|RAM|220GiB||
|SSD|프리미엄 SSD LRS 1TB|  - OPS : 5000 <br>  - 처리량(MBps) : 200 <br>  - 디스크 계층 : P30 |
|CUDA|11.8|12.1 이상의 비추천|
 
 \* 12.1 이상의 CUDA 버전 이용시 겪을 수 있는 일
  - Pytorch3D 미지원으로 인한 [직접 소스코드 빌드](https://github.com/oMFDOo/OpenSourceIssue/issues/7#issue-2681962776)
  - 현재(24.11.22)기준 pytorch-cuda는 12.1까지 이용 가능
  - kaolin의 호환[오류](https://github.com/oMFDOo/OpenSourceIssue/issues/9#issue-2682319760)
  - kaolin 설치 이후 PyTorch 1.9.0 이후 버전 미지원 코드 사용으로 인한 오류 발생
등의 이유로 기존 `12.6`버전에서 `11.8`버전으로 낮추게 되었음.

 \* Azure Virtual Machine [Standard NC24ads A100 v4](https://learn.microsoft.com/ko-kr/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic#host-specifications)을 이용함

 \* 구동 과정 및 분석 과정은 [프로젝트](https://github.com/users/oMFDOo/projects/7)를 이용해 명시하였음

<br>

이하 원문 README.md 내용

<br>

<hr>

## [Paper](https://arxiv.org/abs/2312.03029) | [Project Page](https://yuelangx.github.io/gaussianheadavatar/)
<img src="imgs/teaser.jpg" width="840" height="396"/> 

## Requirements
* Create a conda environment.
```
conda env create -f environment.yaml
```
* Install [Pytorch3d](https://github.com/facebookresearch/pytorch3d).
```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html
```
* Install [kaolin](https://github.com/NVIDIAGameWorks/kaolin).
```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.0_cu113.html
```
* Install diff-gaussian-rasterization and simple_knn from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting). Note, for rendering 32-channel images, please modify "NUM_CHANNELS 3" to "NUM_CHANNELS 32" in "diff-gaussian-rasterization/cuda_rasterizer/config.h".

* 위에 제시한 [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0)의 경로는 현재(2024.12.09)까지 유효한 경로로 나오지만 `simple-knn`에 대해서는 경로가 유효하지 않다. 대안으로 제시하여 구동까지 올바르게 확인된 [simple-knn](https://github.com/DSaurus/simple-knn/tree/main)의 경로는 이것이다. 따라서 제안하는 과정은 다음과 같다.
```
# 제안 다운로드 과정
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization/cuda_rasterizer
vi config.h
### 이곳에서 아래처럼 선언된 값을
### #define NUM_CHANNELS 3 // Default 3, RGB
### 이렇게 바꾼다.
### #define NUM_CHANNELS 32 // Default 3, RGB
cd ../
# pip install 이전 아래가 필요할 수 있다.
sudo apt-get install build-essential
sudo apt-get install libglm-dev
pip install .
cd ../
git clone https://github.com/DSaurus/simple-knn.git
cd simple-knn
pip install .
```
```
cd path/to/gaussian-splatting
# Modify "submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h"
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
* Download ["tets_data.npz"](https://drive.google.com/file/d/1SMkp8v8bDyYxEdyq25jWnAX1zeQuAkNq/view?usp=drive_link) and put it into "assets/".


## Datasets
We provide instructions for preprocessing [NeRSemble dataset](https://tobias-kirschstein.github.io/nersemble/):
* Apply to download [NeRSemble dataset](https://tobias-kirschstein.github.io/nersemble/) and unzip it into "path/to/raw_NeRSemble/".
* Extract the images, cameras and background for specific identities into a structured dataset "NeRSemble/{id}".
```
cd preprocess
python preprocess_nersemble.py
```
* Remove background using [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2). Please git clone the code. Download [pytorch_resnet101.pth](https://drive.google.com/file/d/1zysR-jW6jydA2zkWfevxD1JpQHglKG1_/view?usp=drive_link) and put it into "path/to/BackgroundMattingV2/assets/". Then run the script we provide "preprocess/remove_background_nersemble.py".
```
cp preprocess/remove_background_nersemble.py path/to/BackgroundMattingV2/
cd path/to/BackgroundMattingV2
python remove_background_nersemble.py
```
* Fit BFM model for head pose and expression coefficients using [Multiview-3DMM-Fitting](https://github.com/YuelangX/Multiview-3DMM-Fitting). Please follow the instructions.

We provide a [mini demo dataset](https://drive.google.com/file/d/1OddIml-gJgRQU4YEP-T6USzIQyKSaF7I/view?usp=drive_link) for checking whether the code is runnable. Note, before downloading it, you must first sign the [NeRSemble Terms of Use](https://forms.gle/H4JLdUuehqkBNrBo7).

## Training
First, edit the config file, for example "config/train_meshhead_N031", and train the geometry guidance model.
```
python train_meshhead.py --config config/train_meshhead_N031.yaml
```
Second, edit the config file "config/train_gaussianhead_N031", and train the gaussian head avatar.
```
python train_gaussianhead.py --config config/train_gaussianhead_N031.yaml
```

## Reenactment
Once the two-stage training is completed, the trained avatar can be reenacted by a sequence of expression coefficients. Please specify the avatar checkpoints and the source data in the config file "config/reenactment_N031.py" and run the reenactment application.
```
python reenactment.py --config config/reenactment_N031.yaml
```


## Acknowledgement
Part of the code is borrowed from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Citation
```
@inproceedings{xu2023gaussianheadavatar,
  title={Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians},
  author={Xu, Yuelang and Chen, Benwang and Li, Zhe and Zhang, Hongwen and Wang, Lizhen and Zheng, Zerong and Liu, Yebin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
