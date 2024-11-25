import os
import torch
import argparse

# 프로젝트에서 사용하는 훈련 설정 파일을 로드
from config.config import config_train

# 데이터셋, 데이터 로더, 모듈, 카메라, 기록자 및 훈련 클래스 가져오기
from lib.dataset.Dataset import MeshDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.MeshHeadModule import MeshHeadModule
from lib.module.CameraModule import CameraModule
from lib.recorder.Recorder import MeshHeadTrainRecorder
from lib.trainer.MeshHeadTrainer import MeshHeadTrainer

if __name__ == '__main__':
    # 명령줄 인자를 통해 설정 파일 경로를 받을 수 있도록 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_s1_N031.yaml')  # 기본 설정 파일 경로
    arg = parser.parse_args()

    # 설정 파일 로드 및 구성
    cfg = config_train()  # 설정 객체 초기화
    cfg.load(arg.config)  # 명령줄에서 받은 설정 파일 로드
    cfg = cfg.get_cfg()  # 최종 구성 반환

    # 데이터셋 및 데이터로더 초기화
    dataset = MeshDataset(cfg.dataset)  # 설정 파일 기반으로 데이터셋 객체 생성
    dataloader = DataLoaderX(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)  # 데이터 로딩 방식 설정

    # GPU 장치 설정
    device = torch.device('cuda:%d' % cfg.gpu_id)  # 사용하려는 GPU ID 설정
    torch.cuda.set_device(cfg.gpu_id)  # 설정한 GPU 사용

    # MeshHeadModule 초기화: 초기 랜드마크 데이터를 기반으로 3D 메쉬 모델 생성
    meshhead = MeshHeadModule(cfg.meshheadmodule, dataset.init_landmarks_3d_neutral).to(device)
    if os.path.exists(cfg.load_meshhead_checkpoint):
        # 기존 체크포인트 파일이 존재하면 모델 상태 복구
        meshhead.load_state_dict(torch.load(cfg.load_meshhead_checkpoint, map_location=lambda storage, loc: storage))
    else:
        # 체크포인트가 없으면 구형(pre-trained sphere) 데이터로 사전 훈련 실행
        meshhead.pre_train_sphere(300, device)  # 300 epoch 동안 훈련

    # 카메라 모듈 초기화
    camera = CameraModule()

    # 훈련 진행 중 데이터를 기록하기 위한 기록자 초기화
    recorder = MeshHeadTrainRecorder(cfg.recorder)

    # Adam 옵티마이저로 학습 가능한 매개변수와 학습률 설정
    optimizer = torch.optim.Adam([
        {'params': meshhead.landmarks_3d_neutral, 'lr': cfg.lr_lmk},  # 랜드마크 학습률
        {'params': meshhead.geo_mlp.parameters(), 'lr': cfg.lr_net},  # 기하학적 MLP 학습률
        {'params': meshhead.exp_color_mlp.parameters(), 'lr': cfg.lr_net},  # 표현 색상 MLP 학습률
        {'params': meshhead.pose_color_mlp.parameters(), 'lr': cfg.lr_net},  # 자세 색상 MLP 학습률
        {'params': meshhead.exp_deform_mlp.parameters(), 'lr': cfg.lr_net},  # 표현 변형 MLP 학습률
        {'params': meshhead.pose_deform_mlp.parameters(), 'lr': cfg.lr_net}  # 자세 변형 MLP 학습률
    ])

    # 훈련 객체 초기화: 데이터로더, 메쉬 모델, 카메라, 옵티마이저, 기록자 및 GPU ID를 전달
    trainer = MeshHeadTrainer(dataloader, meshhead, camera, optimizer, recorder, cfg.gpu_id)
    trainer.train(0, 50)  # 훈련 시작 (0부터 50 epoch까지 실행)
