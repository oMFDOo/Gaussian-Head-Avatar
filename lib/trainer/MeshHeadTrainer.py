import torch
import torch.nn.functional as F
from tqdm import tqdm
import kaolin
import lpips
from einops import rearrange
from pytorch3d.transforms import so3_exponential_map

from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance


def laplace_regularizer_const(mesh_verts, mesh_faces):
    """
    라플라스 정규화(Laplacian Regularization)를 적용하여 메쉬의 매끄러움을 유지.
    입력:
        mesh_verts - 메쉬의 정점 좌표 (Nx3)
        mesh_faces - 메쉬의 면 정의 (Fx3)
    출력:
        평균 라플라스 정규화 손실
    """
    term = torch.zeros_like(mesh_verts)  # 정점에 대한 라플라스 결과 초기화
    norm = torch.zeros_like(mesh_verts[..., 0:1])  # 정규화 값 초기화

    # 각 면의 정점을 가져오기
    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    # 각 정점에 대해 면의 기여도 누적
    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1, 3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1, 3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1, 3), (v0 - v2) + (v1 - v2))

    # 각 정점의 면 기여도 계산
    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)  # 면 기여도로 정규화

    return torch.mean(term**2)  # 평균 제곱 손실 반환


class MeshHeadTrainer():
    """
    메쉬 기반 모델을 훈련하는 클래스.
    """
    def __init__(self, dataloader, meshhead, camera, optimizer, recorder, gpu_id):
        """
        클래스 초기화 함수.
        - dataloader: 훈련 데이터를 제공하는 DataLoader 객체.
        - meshhead: 메쉬 모델 객체.
        - camera: 카메라 렌더링 모듈.
        - optimizer: PyTorch 옵티마이저 객체.
        - recorder: 훈련 기록 객체.
        - gpu_id: 사용할 GPU ID.
        """
        self.dataloader = dataloader
        self.meshhead = meshhead
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)  # GPU 설정

    def train(self, start_epoch=0, epochs=1):
        """
        메쉬 모델 훈련 함수.
        - start_epoch: 시작할 에포크 번호.
        - epochs: 훈련할 총 에포크 수.
        """
        for epoch in range(start_epoch, epochs):  # 에포크 반복
            for idx, data in tqdm(enumerate(self.dataloader)):  # 데이터 로드
                # 데이터를 GPU로 이동
                to_cuda = ['images', 'masks', 'visibles', 'intrinsics', 'extrinsics', 'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                # 이미지 및 마스크 데이터를 (B, T, H, W, C)로 변환
                images = data['images'].permute(0, 1, 3, 4, 2)
                masks = data['masks'].permute(0, 1, 3, 4, 2)
                visibles = data['visibles'].permute(0, 1, 3, 4, 2)
                resolution = images.shape[2]  # 이미지 해상도

                # 회전(R), 이동(T), 스케일(S) 적용
                R = so3_exponential_map(data['pose'][:, :3])  # SO(3) 변환 계산
                T = data['pose'][:, 3:, None]  # 평행 이동
                S = data['scale'][:, :, None]  # 스케일
                # 중립 상태 3D 랜드마크 계산
                landmarks_3d_can = (torch.bmm(R.permute(0, 2, 1), (data['landmarks_3d'].permute(0, 2, 1) - T)) / S).permute(0, 2, 1)
                landmarks_3d_neutral = self.meshhead.get_landmarks()[None].repeat(data['landmarks_3d'].shape[0], 1, 1)
                data['landmarks_3d_neutral'] = landmarks_3d_neutral

                # 변형 데이터 생성 및 손실 계산
                deform_data = {
                    'exp_coeff': data['exp_coeff'],  # 표현 계수
                    'query_pts': landmarks_3d_neutral  # 쿼리 점
                }
                deform_data = self.meshhead.deform(deform_data)  # 메쉬 변형
                pred_landmarks_3d_can = deform_data['deformed_pts']  # 변형된 점
                loss_def = F.mse_loss(pred_landmarks_3d_can, landmarks_3d_can)  # MSE 손실 계산

                deform_data = self.meshhead.query_sdf(deform_data)  # Signed Distance Field(SDF) 쿼리
                sdf_landmarks_3d = deform_data['sdf']
                loss_lmk = torch.abs(sdf_landmarks_3d[:, :, 0]).mean()  # 랜드마크 손실 계산

                # 메쉬 재구성 및 렌더링
                data = self.meshhead.reconstruct(data)
                data = self.camera.render(data, resolution)
                render_images = data['render_images']
                render_soft_masks = data['render_soft_masks']
                exp_deform = data['exp_deform']
                pose_deform = data['pose_deform']
                verts_list = data['verts_list']
                faces_list = data['faces_list']

                # 손실 함수 계산
                loss_rgb = F.l1_loss(render_images[:, :, :, :, 0:3] * visibles, images * visibles)  # RGB 손실
                loss_sil = kaolin.metrics.render.mask_iou(
                    (render_soft_masks * visibles[:, :, :, :, 0]).view(-1, resolution, resolution), 
                    (masks * visibles).squeeze().view(-1, resolution, resolution)
                )  # 실루엣 손실
                loss_offset = (exp_deform ** 2).sum(-1).mean() + (pose_deform ** 2).sum(-1).mean()  # 변형 오프셋 손실

                # 라플라스 정규화 손실
                loss_lap = 0.0
                for b in range(len(verts_list)):
                    loss_lap += laplace_regularizer_const(verts_list[b], faces_list[b])

                # 총 손실 계산
                loss = (
                    loss_rgb * 1e-1 + 
                    loss_sil * 1e-1 + 
                    loss_def * 1e0 + 
                    loss_offset * 1e-2 + 
                    loss_lmk * 1e-1 + 
                    loss_lap * 1e2
                )

                # 역전파 및 가중치 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 로그 저장
                log = {
                    'data': data,
                    'meshhead': self.meshhead,
                    'loss_rgb': loss_rgb,
                    'loss_sil': loss_sil,
                    'loss_def': loss_def,
                    'loss_offset': loss_offset,
                    'loss_lmk': loss_lmk,
                    'loss_lap': loss_lap,
                    'epoch': epoch,
                    'iter': idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)  # 기록
