from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
import cv2

# 메쉬 모델 훈련 결과 기록 클래스
class MeshHeadTrainRecorder():
    def __init__(self, cfg):
        """
        MeshHeadTrainRecorder 초기화 함수.
        - cfg: 설정 객체, 로그 디렉토리, 체크포인트 경로, 결과 저장 경로 등을 포함.
        """
        self.logdir = cfg.logdir
        self.logger = SummaryWriter(self.logdir)  # TensorBoard 로그 기록

        self.name = cfg.name
        self.checkpoint_path = cfg.checkpoint_path
        self.result_path = cfg.result_path

        self.save_freq = cfg.save_freq  # 체크포인트 저장 빈도
        self.show_freq = cfg.show_freq  # 결과 이미지를 저장하는 빈도

        # 디렉토리 생성 (존재하지 않을 경우 생성)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

    def log(self, log_data):
        """
        훈련 데이터 로그 기록 및 저장.
        - log_data: 훈련 중 생성된 데이터와 손실 값 포함.
        """
        # 손실 값 기록
        self.logger.add_scalar('loss_rgb', log_data['loss_rgb'], log_data['iter'])
        self.logger.add_scalar('loss_sil', log_data['loss_sil'], log_data['iter'])
        self.logger.add_scalar('loss_def', log_data['loss_def'], log_data['iter'])
        self.logger.add_scalar('loss_offset', log_data['loss_offset'], log_data['iter'])
        self.logger.add_scalar('loss_lmk', log_data['loss_lmk'], log_data['iter'])
        self.logger.add_scalar('loss_lap', log_data['loss_lap'], log_data['iter'])

        # 체크포인트 저장
        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['meshhead'].state_dict(), '%s/%s/meshhead_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['meshhead'].state_dict(), '%s/%s/meshhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))

        # 결과 이미지 저장
        if log_data['iter'] % self.show_freq == 0:
            # 원본 이미지
            image = log_data['data']['images'][0, 0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:, :, ::-1]

            # 렌더링 이미지
            render_image = log_data['data']['render_images'][0, 0, :, :, 0:3].detach().cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:, :, ::-1]

            # 렌더링된 법선 이미지
            render_normal = log_data['data']['render_normals'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
            render_normal = (render_normal * 255).astype(np.uint8)[:, :, ::-1]

            # 크기 조정 및 결과 이미지 저장
            render_image = cv2.resize(render_image, (render_image.shape[0], render_image.shape[1]))
            render_normal = cv2.resize(render_normal, (render_image.shape[0], render_image.shape[1]))
            result = np.hstack((image, render_image, render_normal))  # 이미지들을 가로로 결합
            cv2.imwrite('%s/%s/%06d.jpg' % (self.result_path, self.name, log_data['iter']), result)


# GaussianHead 모델 훈련 결과 기록 클래스
class GaussianHeadTrainRecorder():
    def __init__(self, cfg):
        """
        GaussianHeadTrainRecorder 초기화 함수.
        - cfg: 설정 객체, 로그 디렉토리, 체크포인트 경로, 결과 저장 경로 등을 포함.
        """
        self.logdir = cfg.logdir
        self.logger = SummaryWriter(self.logdir)  # TensorBoard 로그 기록

        self.name = cfg.name
        self.checkpoint_path = cfg.checkpoint_path
        self.result_path = cfg.result_path

        self.save_freq = cfg.save_freq  # 체크포인트 저장 빈도
        self.show_freq = cfg.show_freq  # 결과 이미지를 저장하는 빈도

        # 디렉토리 생성
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

    def log(self, log_data):
        """
        훈련 데이터 로그 기록 및 저장.
        - log_data: 훈련 중 생성된 데이터와 손실 값 포함.
        """
        # 손실 값 기록
        self.logger.add_scalar('loss_rgb_hr', log_data['loss_rgb_hr'], log_data['iter'])
        self.logger.add_scalar('loss_rgb_lr', log_data['loss_rgb_lr'], log_data['iter'])
        self.logger.add_scalar('loss_vgg', log_data['loss_vgg'], log_data['iter'])

        # 체크포인트 저장
        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            torch.save(log_data['supres'].state_dict(), '%s/%s/supres_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['supres'].state_dict(), '%s/%s/supres_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            torch.save(log_data['delta_poses'], '%s/%s/delta_poses_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['delta_poses'], '%s/%s/delta_poses_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))

        # 결과 이미지 저장
        if log_data['iter'] % self.show_freq == 0:
            # 원본 이미지
            image = log_data['data']['images'][0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:, :, ::-1]

            # 렌더링된 이미지
            render_image = log_data['data']['render_images'][0, 0:3].permute(1, 2, 0).detach().cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:, :, ::-1]

            # 잘라낸 이미지
            cropped_image = log_data['data']['cropped_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            cropped_image = (cropped_image * 255).astype(np.uint8)[:, :, ::-1]

            # 초해상도 이미지
            supres_image = log_data['data']['supres_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            supres_image = (supres_image * 255).astype(np.uint8)[:, :, ::-1]

            # 크기 조정 및 결과 이미지 저장
            render_image = cv2.resize(render_image, (image.shape[0], image.shape[1]))
            result = np.hstack((image, render_image, cropped_image, supres_image))  # 이미지들을 가로로 결합
            cv2.imwrite('%s/%s/%06d.jpg' % (self.result_path, self.name, log_data['iter']), result)


# 재현 결과 기록 클래스
class ReenactmentRecorder():
    def __init__(self, cfg):
        """
        ReenactmentRecorder 초기화 함수.
        - cfg: 설정 객체, 결과 저장 경로 등을 포함.
        """
        self.name = cfg.name
        self.result_path = cfg.result_path

        # 결과 디렉토리 생성
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

    def log(self, log_data):
        """
        재현 결과 로그 저장.
        - log_data: 재현 과정에서 생성된 이미지 데이터.
        """
        # 원본 이미지
        image = log_data['data']['images'][0].permute(1, 2, 0).detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)[:, :, ::-1]

        # 초해상도 이미지
        supres_image = log_data['data']['supres_images'][0].permute(1, 2, 0).detach().cpu().numpy()
        supres_image = (supres_image * 255).astype(np.uint8)[:, :, ::-1]

        # 크기 조정 및 결과 이미지 저장
        image = cv2.resize(image, (supres_image.shape[0], supres_image.shape[1]))
        result = np.hstack((image, supres_image))  # 이미지들을 가로로 결합
        cv2.imwrite('%s/%s/%06d.jpg' % (self.result_path, self.name, log_data['iter']), result)
