import os
import torch
import argparse

# 설정 파일과 관련된 모듈 가져오기
from config.config import config_reenactment

# 데이터셋, 데이터 로더, 모듈, 기록자 및 애플리케이션 관련 클래스 가져오기
from lib.dataset.Dataset import ReenactmentDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.GaussianHeadModule import GaussianHeadModule
from lib.module.SuperResolutionModule import SuperResolutionModule
from lib.module.CameraModule import CameraModule
from lib.recorder.Recorder import ReenactmentRecorder
from lib.apps.Reenactment import Reenactment

if __name__ == '__main__':
    # 명령줄 인자를 통해 설정 파일 경로를 받을 수 있도록 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/reenactment_N031.yaml')  # 기본 설정 파일 경로
    arg = parser.parse_args()

    # 설정 파일 로드 및 구성
    cfg = config_reenactment()  # 재현(Reenactment) 작업용 설정 객체 초기화
    cfg.load(arg.config)  # 명령줄에서 받은 설정 파일 로드
    cfg = cfg.get_cfg()  # 최종 구성 반환

    # 데이터셋 및 데이터 로더 초기화
    dataset = ReenactmentDataset(cfg.dataset)  # 설정 파일 기반으로 데이터셋 객체 생성
    dataloader = DataLoaderX(dataset, batch_size=1, shuffle=False, pin_memory=True)  # 배치 크기 1, 섞지 않음

    # GPU 장치 설정
    device = torch.device('cuda:%d' % cfg.gpu_id)  # 사용하려는 GPU ID 설정

    # GaussianHeadModule 초기화: 기존 체크포인트를 로드하여 3D 얼굴 모델 구성
    gaussianhead_state_dict = torch.load(cfg.load_gaussianhead_checkpoint, map_location=lambda storage, loc: storage)
    gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=gaussianhead_state_dict['xyz'],  # 얼굴의 3D 좌표
                                          feature=gaussianhead_state_dict['feature'],  # 얼굴 특징 데이터
                                          landmarks_3d_neutral=gaussianhead_state_dict['landmarks_3d_neutral']).to(device)
    gaussianhead.load_state_dict(gaussianhead_state_dict)  # 모델 가중치 로드

    # SuperResolutionModule 초기화: 체크포인트에서 초해상도(Super Resolution) 모델 로드
    supres = SuperResolutionModule(cfg.supresmodule).to(device)
    supres.load_state_dict(torch.load(cfg.load_supres_checkpoint, map_location=lambda storage, loc: storage))

    # 카메라 모듈 초기화
    camera = CameraModule()

    # 재현 작업의 데이터 기록자 초기화
    recorder = ReenactmentRecorder(cfg.recorder)

    # 재현 애플리케이션 객체 생성
    app = Reenactment(dataloader,        # 데이터 로더 (입력 데이터 제공)
                      gaussianhead,     # GaussianHeadModule (3D 얼굴 모델)
                      supres,           # SuperResolutionModule (고해상도 이미지 생성)
                      camera,           # 카메라 모듈 (렌더링용)
                      recorder,         # 결과 기록용 기록자
                      cfg.gpu_id,       # GPU ID
                      dataset.freeview) # 자유 시점 보기 기능 포함

    # 재현 애플리케이션 실행
    app.run()
