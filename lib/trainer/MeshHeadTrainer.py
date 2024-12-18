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
    term = torch.zeros_like(mesh_verts)  # 정점에 대한 라플라스 결과 초기화, zeros_like : 입력 텐서와 동일한 모양의 텐서를 만들어주는데, 0으로 채워줘
    norm = torch.zeros_like(mesh_verts[..., 0:1])  # 정규화 값 초기화

    # 각 면의 정점을 가져오기
    # - 삼각형 메쉬이기 때문에 하나의 삼각형 면 x, y, z에 대한 정점값을 추산한 것
    # - 기존 입력이 정점 종류, 각 면이 가지는 정점 번호 였기 때문에 이 과정이 필요함
    # - 정점 [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]]
    # - 면 [[0, 1, 2], [1, 2, 3], [1, 3, 2]] 이런 하나의 면은 정점 0, 1, 2번으로 이루어진다에서
    # - 면의 x 모임 [ [0, 1, 0], [1, 1, 0] ] => v0같은 느낌임
    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    # 각 정점에 대해 면의 기여도 누적
    # - 각 x, y, z에 대한 기여도를 계산함
    # - scatter_add_는 인덱스에 해당하는 위치에 값을 더하는 방식으로 작동하여 정점 별로 기여도를 누적함
    # - mesh_faces[:, 0:1].repeat(1, 3)는 그냥 [x]값 뽑고 똑같은 값 3개로 만들어 내는 거 [0, 1, 2], [1, 2, 3] -> [0], [1] -> [0, 0, 0], [1, 1, 1]
    # - 그 다음 v0이 v1, v2에 얼마나 영향을 주는지, v1이 v0, v2에 얼마나 영향을 주는지를 연산함
    # - 0은 그냥 차원임
    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1, 3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1, 3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1, 3), (v0 - v2) + (v1 - v2))

    # 각 정점의 면 기여도 계산 : 각 정점이 얼마나 많은 면에 속하는지를 기록
    # v0와 동일한 크기이면서, 각 원소가 1.0인 텐서를 생성 후 * 2.0을 곱해서, 각 원소 값을 2.0으로 설정
    # - 각 면이 하나의 정점에 대해 기여하는 정도를 2.0으로 설정
    two = torch.ones_like(v0) * 2.0
    # - scatter_add_ : 면의 첫 번째 정점에 대해 면의 기여도 2.0을 누적하는 연산
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    # 각 정점에 대해 계산된 라플라시안 값(term)을 정규화
    # norm : 각 정점이 몇 개의 면에 포함되는지를 나타내는 텐서
    # clamp : norm 값이 최소 1.0이 되도록 클램핑 - 정점이 아예 면에 포함되지 않거나 0인 값이 들어오는 것을 방지
    term = term / torch.clamp(norm, min=1.0)  # 면 기여도로 정규화
    # => 각 정점의 기여도가 면의 수에 의해 불균형하게 커지는 것을 방지하고, 라플라시안 값들이 너무 커지지 않도록 함

    # 최종 손실 계산: 각 정점에 대해 라플라시안 값을 계산하고, 그 제곱의 평균을 반환
    # Loss = (1 / N) * Σ (Δv_i)^2
    # 여기서 Δv_i는 각 정점 v_i에 대한 라플라시안 결과로, 이웃 정점들과의 위치 차이를 나타냄
    # 이 손실은 메쉬의 매끄러움을 유지하기 위해, 각 정점이 이웃 정점들과 얼마나 균형 있게 배치되어 있는지를 측정
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
                # images: 모델에 입력되는 이미지 데이터
                # masks: 이미지에 대한 마스크 정보
                # visibles: 가시성 정보 (예: 이미지에서 보이는 부분)
                # intrinsics: 카메라 내부 파라미터 (내부 행렬 등)
                # extrinsics: 카메라 외부 파라미터 (위치, 방향 등)
                # pose: 사람 또는 객체의 포즈 정보
                # scale: 크기 정보
                # exp_coeff: 표정 계수 (facial expression coefficients)
                # landmarks_3d: 3D 랜드마크
                # exp_id: 표정 식별자
                to_cuda = ['images', 'masks', 'visibles', 'intrinsics', 'extrinsics', 'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                # 이미지 및 마스크 데이터를 원래 (B, C, H, W)의 형태에서 (B, T, H, W, C)로 바꾸는 작업
                # permute : 텐서의 차원 순서를 변경하는 함수
                # => 모델이 데이터의 특정 차원을 시간적 차원이나 다른 특성으로 해석할 수 있게 하기 위함
                # 0: 배치 크기 (B)
                # 1: 시간 차원 (T) 또는 다른 차원 (e.g., 시퀀스 길이)
                # 3: 이미지 높이 (H)
                # 4: 이미지 너비 (W)
                # 2: 채널 (C)
                images = data['images'].permute(0, 1, 3, 4, 2)
                masks = data['masks'].permute(0, 1, 3, 4, 2)
                visibles = data['visibles'].permute(0, 1, 3, 4, 2)
                resolution = images.shape[2]  # 이미지 해상도, H 추출

                # 회전(R), 이동(T), 스케일(S) 적용
                # data['pose']는 3D 포즈 정보를 담고 있는 텐서로, 일반적으로 사람의 위치와 방향을 나타냄
                # 이 포즈 정보는 보통 6D 벡터로 표현되며, 앞의 3개 값은 rotation, 뒤의 3개 값은 translation을 의미함
                # data['pose'][:, :3]는 회전 정보만 추출한 것
                # SO(3) - Special Orthogonal Group 3D는 3D 공간에서의 회전 그룹. 이 변환을 통해서 회전 행렬로 변환해 3D 공간에서 회전 변환을 적용함
                # => 회전 상태의 객체/카메라 학습을 위함
                R = so3_exponential_map(data['pose'][:, :3])  # SO(3) 변환 계산
                T = data['pose'][:, 3:, None]  # 평행 이동
                S = data['scale'][:, :, None]  # 스케일
                # => 일반화 능력 향상과 모델의 강건성(Robustness)을 높이기 위함
                
                # 중립 표정(Neutral landmarks) 계산 및 3D 랜드마크 변환
                # landmarks_3d - T: 랜드마크에서 이동 벡터(T)를 빼면, 3D 공간에서 원점에 대한 상대적인 위치를 구해 이동 변환을 적용
                # R.permute(0, 2, 1): 회전 행렬 R을 사용하여 랜드마크를 회전
                # torch.bmm는 배치 행렬 곱셈을 수행, 회전 행렬을 랜드마크의 상대 위치에 곱해줌으로써 회전된 랜드마크 위치를 계산
                # => landmarks_3d_can : 3D 랜드마크가 실제 포즈나 표정 변형에 맞춰 변환된 결과
                landmarks_3d_can = (torch.bmm(R.permute(0, 2, 1), (data['landmarks_3d'].permute(0, 2, 1) - T)) / S).permute(0, 2, 1)
                # => andmarks_3d_neutral : 기본 중립 표정에서의 3D 랜드마크
                landmarks_3d_neutral = self.meshhead.get_landmarks()[None].repeat(data['landmarks_3d'].shape[0], 1, 1)
                data['landmarks_3d_neutral'] = landmarks_3d_neutral

                # 변형 데이터 생성 및 손실 계산
                deform_data = {
                    'exp_coeff': data['exp_coeff'],  # 표현 계수
                    'query_pts': landmarks_3d_neutral  # 쿼리 점 : 중립 표정에서의 3D 랜드마크
                }
                # 중립 표정에서의 3D 랜드마크를 표정 변화에 맞게 변형
                deform_data = self.meshhead.deform(deform_data)  # 메쉬 변형
                # pred_landmarks_3d_can : 변형된 랜드마크
                # deformed_pts : 메쉬 변형을 통해 예측된 3D 랜드마크 위치
                pred_landmarks_3d_can = deform_data['deformed_pts']  # 변형된 점
                # 랜드마크 loss 추정
                loss_def = F.mse_loss(pred_landmarks_3d_can, landmarks_3d_can)  # MSE 손실 계산

                # query_sdf를 통해 변형된 랜드마크들이 3D 표면과 얼마나 가까운지 측정
                deform_data = self.meshhead.query_sdf(deform_data)  # Signed Distance Field(SDF) 쿼리
                sdf_landmarks_3d = deform_data['sdf']
                # SDF 값이 0에 가까울수록 랜드마크가 객체의 표면에 가까운 것이므로, 손실 값이 작은 값
                loss_lmk = torch.abs(sdf_landmarks_3d[:, :, 0]).mean()  # 랜드마크 손실 계산

                # 메쉬 재구성 및 렌더링
                # 표현 계수(exp_coeff), 표정 변화, 포즈 변화 등의 정보들을 바탕으로 3D 메쉬 모델을 다시 생성
                data = self.meshhead.reconstruct(data) 
                # 카메라 모델이 데이터를 사용해 2D 이미지를 생성
                data = self.camera.render(data, resolution)
                ####################################### 최종 결과물 #######################################
                # ****모델이 생성한 최종 이미지****
                render_images = data['render_images']
                # 각 픽셀에 대해 확률적인 값을 가진 소프트마스크로, 각 픽셀이 얼굴/객체의 일부일 확률을 표현
                render_soft_masks = data['render_soft_masks']
                # 표정 변화에 따른 3D 메쉬 변형을 나타내는 값
                exp_deform = data['exp_deform']
                # 얼굴이나 객체의 포즈 변화에 따른 3D 메쉬 변형을 나타내는 값
                pose_deform = data['pose_deform']
                # 3D 메쉬의 각 정점 좌표를 나열한 리스트로, 각 정점은 3D 공간에서 객체의 형태를 정의
                verts_list = data['verts_list']
                # 3D 메쉬의 면 리스트
                faces_list = data['faces_list']

                ####################################### 손실 함수 계산 #######################################
                # RGB 손실
                # [:, :, :, :, 0:3] : rgb 3개 채널에 대해서만 예측 이미지 render_images와 실제 이미지 loss 추정
                # visibles : 보이는 영역을 나타낸 마스크, 객체가 가려져 있거나 일부가 잘리면 해당 영역은 0, 나머지는 1로 설정
                loss_rgb = F.l1_loss(render_images[:, :, :, :, 0:3] * visibles, images * visibles)  
                # 실루엣 손실
                # Intersection over Union (IoU) : 두 마스크 간의 겹치는 영역의 비율을 측정하는 지표
                loss_sil = kaolin.metrics.render.mask_iou(
                    (render_soft_masks * visibles[:, :, :, :, 0]).view(-1, resolution, resolution), 
                    (masks * visibles).squeeze().view(-1, resolution, resolution)
                ) 
                # 변형 오프셋 손실
                # 표정과 포즈의 변형 오프셋에 대한 합산 손실, 표정과 포즈에 따른 과도한 변형을 최소화하기 위함
                loss_offset = (exp_deform ** 2).sum(-1).mean() + (pose_deform ** 2).sum(-1).mean()  

                # 라플라스 정규화 손실
                # 모든 배치에 대해 라플라스 정규화 손실을 합산
                loss_lap = 0.0
                for b in range(len(verts_list)):
                    loss_lap += laplace_regularizer_const(verts_list[b], faces_list[b])

                # 총 손실 계산
                
                loss = (
                    loss_rgb * 1e-1 +  # RGB 손실에 가중치 0.1 적용
                    loss_sil * 1e-1 +  # 실루엣 손실에 가중치 0.1 적용
                    loss_def * 1e0 +   # 변형 손실에 가중치 1 적용 (가장 중요한 손실)
                    loss_offset * 1e-2 +  # 오프셋 손실에 가중치 0.01 적용
                    loss_lmk * 1e-1 +  # 랜드마크 손실에 가중치 0.1 적용
                    loss_lap * 1e2     # 라플라시안 손실에 가중치 100 적용 (가장 중요한 손실)
                )

                # 역전파 및 가중치 업데이트
                self.optimizer.zero_grad()  # 옵티마이저의 기울기 초기화 (기존 기울기 값이 남아있는 것을 방지)
                loss.backward()  # 손실 함수에 대해 역전파 수행 (기울기 계산)
                self.optimizer.step()  # 옵티마이저를 사용하여 가중치 업데이트

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
