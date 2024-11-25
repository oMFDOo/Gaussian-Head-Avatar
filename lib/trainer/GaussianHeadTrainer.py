import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import lpips

class GaussianHeadTrainer():
    """
    GaussianHeadTrainer 클래스는 GaussianHead 모델을 훈련하는 데 사용됩니다.
    """
    def __init__(self, dataloader, delta_poses, gaussianhead, supres, camera, optimizer, recorder, gpu_id):
        """
        클래스 초기화 함수.
        - dataloader: 훈련 데이터를 제공하는 DataLoader 객체.
        - delta_poses: 자세 보정 데이터.
        - gaussianhead: GaussianHead 모델 객체.
        - supres: Super Resolution 모듈 객체.
        - camera: 카메라 모듈 객체.
        - optimizer: PyTorch 옵티마이저 객체.
        - recorder: 훈련 기록을 저장하는 객체.
        - gpu_id: 사용할 GPU ID.
        """
        self.dataloader = dataloader
        self.delta_poses = delta_poses
        self.gaussianhead = gaussianhead
        self.supres = supres
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)  # GPU 설정
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)  # VGG 기반 LPIPS 손실 함수 초기화
        self.Ld_value = 0.5  # 임시로 설정한 값 (추후 조정 가능)

    def train(self, start_epoch=0, epochs=1):
        """
        훈련을 실행하는 함수.
        - start_epoch: 훈련을 시작할 에포크 번호.
        - epochs: 총 훈련 에포크 수.
        """
        for epoch in range(start_epoch, epochs):  # 에포크 반복
            for idx, data in tqdm(enumerate(self.dataloader)):  # 데이터 로드 및 진행 표시
                # 데이터를 GPU로 이동
                to_cuda = ['images', 'masks', 'visibles', 'images_coarse', 'masks_coarse', 'visibles_coarse', 
                           'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                           'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                # 원본 및 가시성 데이터 준비
                images = data['images']
                visibles = data['visibles']
                if self.supres is None:
                    # 초해상도 모듈이 없는 경우, 원본 이미지를 사용
                    images_coarse = images
                    visibles_coarse = visibles
                else:
                    # 초해상도 모듈이 있는 경우, 저해상도 이미지를 사용
                    images_coarse = data['images_coarse']
                    visibles_coarse = data['visibles_coarse']

                resolution_coarse = images_coarse.shape[2]  # 저해상도 이미지의 해상도
                resolution_fine = images.shape[2]  # 고해상도 이미지의 해상도

                # 자세 보정 데이터 적용
                data['pose'] = data['pose'] + self.delta_poses[data['exp_id'], :]

                # 저해상도 이미지 생성
                data = self.gaussianhead.generate(data)
                data = self.camera.render_gaussian(data, resolution_coarse)
                render_images = data['render_images']

                # 이미지를 무작위 크기로 잘라 데이터 증강 수행
                scale_factor = random.random() * 0.45 + 0.8  # 크기 조정 비율
                scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
                cropped_render_images, cropped_images, cropped_visibles = self.random_crop(render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine)
                data['cropped_images'] = cropped_images

                # 초해상도 이미지를 생성
                supres_images = self.supres(cropped_render_images)
                data['supres_images'] = supres_images

                # 손실 함수 계산
                # 저해상도 이미지 손실 (L1 손실)
                loss_rgb_lr = F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)
                # 고해상도 이미지 손실 (L1 손실)
                loss_rgb_hr = F.l1_loss(supres_images * cropped_visibles, cropped_images * cropped_visibles)
                # VGG 기반 LPIPS 손실 계산 (일부 영역에서만)
                left_up = (random.randint(0, supres_images.shape[2] - 512), random.randint(0, supres_images.shape[3] - 512))
                loss_vgg = self.fn_lpips((supres_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], 
                                         (cropped_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], normalize=True).mean()
                # 총 손실
                loss = loss_rgb_hr + loss_rgb_lr + loss_vgg * 1e-1

                # 역전파 및 가중치 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 로그 저장
                log = {
                    'data': data,
                    'delta_poses': self.delta_poses,
                    'gaussianhead': self.gaussianhead,
                    'supres': self.supres,
                    'loss_rgb_lr': loss_rgb_lr,
                    'loss_rgb_hr': loss_rgb_hr,
                    'loss_vgg': loss_vgg,
                    'epoch': epoch,
                    'iter': idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)

    def random_crop(self, render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine):
        """
        이미지를 무작위 크기로 잘라 데이터 증강을 수행.
        - render_images: 생성된 렌더링 이미지
        - images: 원본 이미지
        - visibles: 가시성 데이터
        - scale_factor: 크기 조정 비율
        - resolution_coarse: 저해상도 이미지 해상도
        - resolution_fine: 고해상도 이미지 해상도
        반환값:
            잘라낸 렌더링 이미지, 원본 이미지, 가시성 데이터
        """
        # 크기 조정
        render_images_scaled = F.interpolate(render_images, scale_factor=scale_factor)
        images_scaled = F.interpolate(images, scale_factor=scale_factor)
        visibles_scaled = F.interpolate(visibles, scale_factor=scale_factor)

        if scale_factor < 1:
            # 크기 조정 후 이미지를 채우는 경우
            render_images = torch.ones([render_images_scaled.shape[0], render_images_scaled.shape[1], resolution_coarse, resolution_coarse], device=self.device)
            left_up_coarse = (random.randint(0, resolution_coarse - render_images_scaled.shape[2]), random.randint(0, resolution_coarse - render_images_scaled.shape[3]))
            render_images[:, :, left_up_coarse[0]: left_up_coarse[0] + render_images_scaled.shape[2], left_up_coarse[1]: left_up_coarse[1] + render_images_scaled.shape[3]] = render_images_scaled

            images = torch.ones([images_scaled.shape[0], images_scaled.shape[1], resolution_fine, resolution_fine], device=self.device)
            visibles = torch.ones([visibles_scaled.shape[0], visibles_scaled.shape[1], resolution_fine, resolution_fine], device=self.device)
            left_up_fine = (int(left_up_coarse[0] * resolution_fine / resolution_coarse), int(left_up_coarse[1] * resolution_fine / resolution_coarse))
            images[:, :, left_up_fine[0]: left_up_fine[0] + images_scaled.shape[2], left_up_fine[1]: left_up_fine[1] + images_scaled.shape[3]] = images_scaled
            visibles[:, :, left_up_fine[0]: left_up_fine[0] + visibles_scaled.shape[2], left_up_fine[1]: left_up_fine[1] + visibles_scaled.shape[3]] = visibles_scaled
        else:
            # 크기 조정 후 자르는 경우
            left_up_coarse = (random.randint(0, render_images_scaled.shape[2] - resolution_coarse), random.randint(0, render_images_scaled.shape[3] - resolution_coarse))
            render_images = render_images_scaled[:, :, left_up_coarse[0]: left_up_coarse[0] + resolution_coarse, left_up_coarse[1]: left_up_coarse[1] + resolution_coarse]

            left_up_fine = (int(left_up_coarse[0] * resolution_fine / resolution_coarse), int(left_up_coarse[1] * resolution_fine / resolution_coarse))
            images = images_scaled[:, :, left_up_fine[0]: left_up_fine[0] + resolution_fine, left_up_fine[1]: left_up_fine[1] + resolution_fine]
            visibles = visibles_scaled[:, :, left_up_fine[0]: left_up_fine[0] + resolution_fine, left_up_fine[1]: left_up_fine[1] + resolution_fine]
        
        return render_images, images, visibles
