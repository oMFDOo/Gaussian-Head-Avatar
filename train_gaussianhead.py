import os
import torch
import argparse

# 프로젝트에서 사용되는 훈련 설정 로드
from config.config import config_train

# 데이터셋, 데이터로더, 모듈, 훈련 및 기록 관련 클래스 가져오기
from lib.dataset.Dataset import GaussianDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.MeshHeadModule import MeshHeadModule
from lib.module.GaussianHeadModule import GaussianHeadModule
from lib.module.SuperResolutionModule import SuperResolutionModule
from lib.module.CameraModule import CameraModule
from lib.recorder.Recorder import GaussianHeadTrainRecorder
from lib.trainer.GaussianHeadTrainer import GaussianHeadTrainer

if __name__ == '__main__':
    # 명령줄 인자 설정 (예: 설정 파일 경로를 입력받음)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_s2_N031.yaml')
    arg = parser.parse_args()

    # 훈련 설정 파일 로드 및 구성
    cfg = config_train()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    # 데이터셋 및 데이터로더 초기화
    dataset = GaussianDataset(cfg.dataset)  # 설정 파일 기반으로 데이터셋 생성
    dataloader = DataLoaderX(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)  # 데이터 배치 로드 설정

    # GPU 장치 설정
    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    # GaussianHeadModule 초기화: 기존 체크포인트 로드 또는 새로운 데이터 생성
    if os.path.exists(cfg.load_gaussianhead_checkpoint):
        # 체크포인트 파일 로드 (기존 모델 상태 복구)
        gaussianhead_state_dict = torch.load(cfg.load_gaussianhead_checkpoint, map_location=lambda storage, loc: storage)
        gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=gaussianhead_state_dict['xyz'], 
                                          feature=gaussianhead_state_dict['feature'],
                                          landmarks_3d_neutral=gaussianhead_state_dict['landmarks_3d_neutral']).to(device)
        gaussianhead.load_state_dict(gaussianhead_state_dict)
    else:
        # MeshHeadModule로 초기 데이터 생성
        meshhead_state_dict = torch.load(cfg.load_meshhead_checkpoint, map_location=lambda storage, loc: storage)
        meshhead = MeshHeadModule(cfg.meshheadmodule, meshhead_state_dict['landmarks_3d_neutral']).to(device)
        meshhead.load_state_dict(meshhead_state_dict)
        meshhead.subdivide()  # 메쉬 세분화
        with torch.no_grad():
            data = meshhead.reconstruct_neutral()  # 중립 상태 재구성

        # GaussianHeadModule을 초기화하여 모델 정의
        gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=data['verts'].cpu(),
                                          feature=torch.atanh(data['verts_feature'].cpu()), 
                                          landmarks_3d_neutral=meshhead.landmarks_3d_neutral.detach().cpu(),
                                          add_mouth_points=True).to(device)
        # 다른 MLP 상태 복사
        gaussianhead.exp_color_mlp.load_state_dict(meshhead.exp_color_mlp.state_dict())
        gaussianhead.pose_color_mlp.load_state_dict(meshhead.pose_color_mlp.state_dict())
        gaussianhead.exp_deform_mlp.load_state_dict(meshhead.exp_deform_mlp.state_dict())
        gaussianhead.pose_deform_mlp.load_state_dict(meshhead.pose_deform_mlp.state_dict())
    
    # SuperResolutionModule 초기화 (해상도 업스케일링)
    supres = SuperResolutionModule(cfg.supresmodule).to(device)
    if os.path.exists(cfg.load_supres_checkpoint):
        supres.load_state_dict(torch.load(cfg.load_supres_checkpoint, map_location=lambda storage, loc: storage))

    # 카메라 모듈 및 기록자 초기화
    camera = CameraModule()
    recorder = GaussianHeadTrainRecorder(cfg.recorder)

    # 모델 최적화를 위한 학습 가능한 매개변수 지정
    optimized_parameters = [{'params' : supres.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.xyz, 'lr' : cfg.lr_net * 0.1},
                            {'params' : gaussianhead.feature, 'lr' : cfg.lr_net * 0.1},
                            {'params' : gaussianhead.exp_color_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.pose_color_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.exp_deform_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.pose_deform_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.exp_attributes_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.pose_attributes_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.scales, 'lr' : cfg.lr_net * 0.3},
                            {'params' : gaussianhead.rotation, 'lr' : cfg.lr_net * 0.1},
                            {'params' : gaussianhead.opacity, 'lr' : cfg.lr_net}]
    
    # 자세(포즈) 보정 데이터 로드 또는 초기화
    if os.path.exists(cfg.load_delta_poses_checkpoint):
        delta_poses = torch.load(cfg.load_delta_poses_checkpoint)
    else:
        delta_poses = torch.zeros([dataset.num_exp_id, 6]).to(device)

    # 자세 데이터 최적화 여부 확인
    if cfg.optimize_pose:
        delta_poses = delta_poses.requires_grad_(True)  # 학습 가능한 변수로 설정
        optimized_parameters.append({'params' : delta_poses, 'lr' : cfg.lr_pose})
    else:
        delta_poses = delta_poses.requires_grad_(False)

    # Adam 옵티마이저 초기화
    optimizer = torch.optim.Adam(optimized_parameters)

    # 훈련 프로세스 시작
    trainer = GaussianHeadTrainer(dataloader, delta_poses, gaussianhead, supres, camera, optimizer, recorder, cfg.gpu_id)
    trainer.train(0, 1000)  # 최대 1000 epoch까지 훈련
