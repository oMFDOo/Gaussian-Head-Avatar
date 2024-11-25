import torch
from torch import nn
import numpy as np
import kaolin
import tqdm
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.transforms import so3_exponential_map

from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder
from lib.utils.dmtet_utils import marching_tetrahedra


class MeshHeadModule(nn.Module):
    """
    MeshHeadModule 클래스:
    - 이 클래스는 3D 메쉬를 생성하고 변형하며, 표정과 자세에 따라 색상과 변형을 제어합니다.
    - '머리 메쉬'라고 불리는 정점과 면(face)의 데이터를 기반으로 작동합니다.
    """

    def __init__(self, cfg, init_landmarks_3d_neutral):
        """
        초기화 함수:
        - cfg: 모델 구성(configuration) 정보
        - init_landmarks_3d_neutral: 중립적인 상태의 초기 랜드마크(3D 점)

        주요 역할:
        1. MLP(다층 퍼셉트론) 네트워크 초기화
        2. 랜드마크, 테트라헤드(tetrahedra) 및 기타 파라미터 초기화
        """
        super(MeshHeadModule, self).__init__()

        # 1. 여러 MLP(다층 퍼셉트론) 정의: 기하학, 색상, 변형을 계산하기 위해 사용
        self.geo_mlp = MLP(cfg.geo_mlp, last_op=nn.Tanh())  # 기하학(Geometry) 관련 MLP
        self.exp_color_mlp = MLP(cfg.exp_color_mlp, last_op=nn.Sigmoid())  # 표정 기반 색상
        self.pose_color_mlp = MLP(cfg.pose_color_mlp, last_op=nn.Sigmoid())  # 자세 기반 색상
        self.exp_deform_mlp = MLP(cfg.exp_deform_mlp, last_op=nn.Tanh())  # 표정 기반 변형
        self.pose_deform_mlp = MLP(cfg.pose_deform_mlp, last_op=nn.Tanh())  # 자세 기반 변형

        # 2. 초기 랜드마크(학습 가능 매개변수로 설정)
        self.landmarks_3d_neutral = nn.Parameter(init_landmarks_3d_neutral)

        # 3. 위치 임베딩(Positional Embedding) 생성
        self.pos_embedding, _ = get_embedder(cfg.pos_freq)

        # 4. 모델의 공간적 범위 및 거리 관련 설정값
        self.model_bbox = cfg.model_bbox  # 메쉬가 포함될 범위
        self.dist_threshold_near = cfg.dist_threshold_near  # 가까운 거리 임계값
        self.dist_threshold_far = cfg.dist_threshold_far  # 먼 거리 임계값
        self.deform_scale = cfg.deform_scale  # 변형의 스케일 조정값

        # 5. 테트라헤드(Tetrahedra) 데이터를 로드하여 모델에 등록
        tets_data = np.load('assets/tets_data.npz')  # 사전 정의된 테트라헤드 데이터 로드
        self.register_buffer('tet_verts', torch.from_numpy(tets_data['tet_verts']))  # 정점 데이터
        self.register_buffer('tets', torch.from_numpy(tets_data['tets']))  # 테트라헤드 데이터
        self.grid_res = 128  # 초기 해상도 설정

        # 6. 테트라헤드 분할 여부
        if cfg.subdivide:
            self.subdivide()  # 테트라헤드 정점 및 면 분할

    def geometry(self, geo_input):
        """
        기하학 MLP를 통해 SDF(Signed Distance Field)를 예측합니다.
        - geo_input: 위치 임베딩된 입력 (BxNxD 형태)
        반환값: 예측 결과 (BxNx1 형태)
        """
        pred = self.geo_mlp(geo_input)
        return pred

    def exp_color(self, color_input):
        """
        표정 기반 색상을 예측합니다.
        - color_input: 표정과 관련된 특징 벡터
        반환값: 정점의 색상 값
        """
        verts_color = self.exp_color_mlp(color_input)
        return verts_color

    def pose_color(self, color_input):
        """
        자세 기반 색상을 예측합니다.
        - color_input: 자세와 관련된 특징 벡터
        반환값: 정점의 색상 값
        """
        verts_color = self.pose_color_mlp(color_input)
        return verts_color

    def exp_deform(self, deform_input):
        """
        표정 기반 변형을 예측합니다.
        - deform_input: 표정과 관련된 특징 벡터
        반환값: 정점의 변형 벡터
        """
        deform = self.exp_deform_mlp(deform_input)
        return deform

    def pose_deform(self, deform_input):
        """
        자세 기반 변형을 예측합니다.
        - deform_input: 자세와 관련된 특징 벡터
        반환값: 정점의 변형 벡터
        """
        deform = self.pose_deform_mlp(deform_input)
        return deform

    def get_landmarks(self):
        """
        현재 상태의 랜드마크를 반환합니다.
        반환값: 3D 랜드마크 좌표
        """
        return self.landmarks_3d_neutral

    def subdivide(self):
        """
        테트라헤드(tetrahedra)를 더 세밀하게 분할합니다.
        - 테트라헤드를 분할하여 해상도를 높이고, 정교한 결과를 얻습니다.
        """
        new_tet_verts, new_tets = kaolin.ops.mesh.subdivide_tetmesh(self.tet_verts.unsqueeze(0), self.tets)
        self.tet_verts = new_tet_verts[0]
        self.tets = new_tets
        self.grid_res *= 2  # 해상도 2배 증가

    def reconstruct(self, data):
        """
        3D 메쉬를 재구성하는 함수:
        - data: 표정, 자세, 크기(scale) 등의 입력 데이터.
        - 텐서의 정점(verts), 면(faces)을 생성하고 변형 및 색상을 적용합니다.
        """
        B = data['exp_coeff'].shape[0]  # 배치 크기(B)

        # 테트라헤드 정점을 복제하여 배치 크기에 맞게 확장
        query_pts = self.tet_verts.unsqueeze(0).repeat(B, 1, 1)

        # 각 정점의 임베딩 벡터 계산
        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        # MLP를 사용해 기하학적 정보 예측 (SDF, 변형, 특징)
        pred = self.geometry(geo_input)
        sdf, deform, features = pred[:, :1, :], pred[:, 1:4, :], pred[:, 4:, :]
        sdf = sdf.permute(0, 2, 1)
        features = features.permute(0, 2, 1)

        # 정점에 변형(deform)을 적용
        verts_deformed = (query_pts + torch.tanh(deform.permute(0, 2, 1)) / self.grid_res)

        # Tetrahedra 메쉬를 3D 표면으로 변환
        verts_list, features_list, faces_list = marching_tetrahedra(verts_deformed, features, self.tets, sdf)

        # 데이터에 결과 저장
        data['verts0_list'] = verts_list
        data['faces_list'] = faces_list

        # 정점 배치 처리
        verts_batch = []
        verts_features_batch = []
        num_pts_max = 0  # 최대 정점 수
        for b in range(B):
            if verts_list[b].shape[0] > num_pts_max:
                num_pts_max = verts_list[b].shape[0]

        for b in range(B):
            verts_batch.append(torch.cat([verts_list[b], torch.zeros([num_pts_max - verts_list[b].shape[0], verts_list[b].shape[1]], device=verts_list[b].device)], 0))
            verts_features_batch.append(torch.cat([features_list[b], torch.zeros([num_pts_max - features_list[b].shape[0], features_list[b].shape[1]], device=features_list[b].device)], 0))

        verts_batch = torch.stack(verts_batch, 0)
        verts_features_batch = torch.stack(verts_features_batch, 0)

        # 거리 기반으로 표정/자세 가중치 계산
        dists, idx, _ = knn_points(verts_batch, data['landmarks_3d_neutral'])
        exp_weights = torch.clamp((self.dist_threshold_far - dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)
        pose_weights = 1 - exp_weights

        # 표정과 자세 기반으로 색상 계산
        exp_color_input = torch.cat([verts_features_batch.permute(0, 2, 1), data['exp_coeff'].unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        verts_color_batch = self.exp_color(exp_color_input).permute(0, 2, 1) * exp_weights

        pose_color_input = torch.cat([verts_features_batch.permute(0, 2, 1), self.pos_embedding(data['pose']).unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        verts_color_batch = verts_color_batch + self.pose_color(pose_color_input).permute(0, 2, 1) * pose_weights

        # 데이터에 색상 결과 저장
        data['verts_color_list'] = [verts_color_batch[b, :verts_list[b].shape[0], :] for b in range(B)]
        return data

    def pre_train_sphere(self, iter, device):
        """
        MLP를 구체(sphere)를 기준으로 사전 학습합니다.
        - iter: 학습 반복 횟수
        - device: 학습에 사용할 장치 (GPU/CPU)
        """
        loss_fn = torch.nn.MSELoss()  # 손실 함수 정의
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)  # Adam 최적화

        for i in tqdm.tqdm(range(iter)):
            # 구체 표면에서 랜덤 점 생성
            query_pts = torch.rand((8, 1024, 3), device=device) * 3 - 1.5
            ref_value = torch.sqrt((query_pts ** 2).sum(-1)) - 1.0  # 구체 표면까지의 거리

            # SDF 예측
            data = {'query_pts': query_pts}
            data = self.query_sdf(data)
            sdf = data['sdf']

            # 손실 계산 및 최적화
            loss = loss_fn(sdf[:, :, 0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Pre-trained MLP", loss.item())
