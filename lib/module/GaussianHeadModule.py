import torch
from torch import nn
from einops import rearrange
import tqdm
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from simple_knn._C import distCUDA2

from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder
from lib.utils.general_utils import inverse_sigmoid


class GaussianHeadModule(nn.Module):
    """
    GaussianHeadModule:
    - 3D 정점(xyz)과 해당 특징(feature)을 입력받아 변형(deformation), 색상(color), 스케일(scale),
      회전(rotation), 불투명도(opacity)을 모델링.
    """

    def __init__(self, cfg, xyz, feature, landmarks_3d_neutral, add_mouth_points=False):
        """
        초기화 함수:
        - cfg: 설정 값
        - xyz: 초기 3D 정점 좌표
        - feature: 각 정점의 특징 벡터
        - landmarks_3d_neutral: 중립적인 얼굴 랜드마크
        - add_mouth_points: 입 주변의 추가 정점 여부
        """
        super(GaussianHeadModule, self).__init__()

        if add_mouth_points and cfg.num_add_mouth_points > 0:
            # 입 주변의 추가 정점을 생성
            mouth_keypoints = landmarks_3d_neutral[48:66]  # 입 주변 랜드마크 추출
            mouth_center = torch.mean(mouth_keypoints, dim=0, keepdim=True)  # 입의 중심점 계산
            mouth_center[:, 2] = mouth_keypoints[:, 2].min()  # 입 깊이 조정
            max_dist = (mouth_keypoints - mouth_center).abs().max(0)[0]  # 입 주변의 최대 거리 계산
            points_add = (torch.rand([cfg.num_add_mouth_points, 3]) - 0.5) * 1.6 * max_dist + mouth_center  # 임의의 추가 정점 생성

            xyz = torch.cat([xyz, points_add])  # 추가 정점을 xyz에 결합
            feature = torch.cat([feature, torch.zeros([cfg.num_add_mouth_points, feature.shape[1]])])  # 해당 정점의 특징 초기화

        # 정점 좌표(xyz)와 특징(feature)을 nn.Parameter로 변환 (학습 가능)
        self.xyz = nn.Parameter(xyz)
        self.feature = nn.Parameter(feature)
        self.register_buffer('landmarks_3d_neutral', landmarks_3d_neutral)  # 랜드마크를 고정된 버퍼로 등록

        # 정점 간 거리 계산 및 초기 스케일 설정
        dist2 = torch.clamp_min(distCUDA2(self.xyz.cuda()), 0.0000001).cpu()  # 정점 간 거리 계산 (CUDA 사용)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)  # 초기 스케일 설정 (로그 스케일)
        self.scales = nn.Parameter(scales)

        # 회전 파라미터 초기화 (단위 쿼터니언)
        rots = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        rots[:, 0] = 1  # 단위 쿼터니언 (회전 없음)
        self.rotation = nn.Parameter(rots)

        # 불투명도(opacity) 초기화 (0.3에서 시작)
        self.opacity = nn.Parameter(inverse_sigmoid(0.3 * torch.ones((xyz.shape[0], 1))))

        # MLP(다층 퍼셉트론) 네트워크 정의
        self.exp_color_mlp = MLP(cfg.exp_color_mlp, last_op=None)  # 표정 기반 색상 MLP
        self.pose_color_mlp = MLP(cfg.pose_color_mlp, last_op=None)  # 자세 기반 색상 MLP
        self.exp_attributes_mlp = MLP(cfg.exp_attributes_mlp, last_op=None)  # 표정 기반 속성 MLP
        self.pose_attributes_mlp = MLP(cfg.pose_attributes_mlp, last_op=None)  # 자세 기반 속성 MLP
        self.exp_deform_mlp = MLP(cfg.exp_deform_mlp, last_op=nn.Tanh())  # 표정 기반 변형 MLP
        self.pose_deform_mlp = MLP(cfg.pose_deform_mlp, last_op=nn.Tanh())  # 자세 기반 변형 MLP

        # 위치 임베딩(Positional Embedding) 생성
        self.pos_embedding, _ = get_embedder(cfg.pos_freq)

        # 기타 설정 값
        self.exp_coeffs_dim = cfg.exp_coeffs_dim  # 표정 계수 차원
        self.dist_threshold_near = cfg.dist_threshold_near  # 가까운 거리 임계값
        self.dist_threshold_far = cfg.dist_threshold_far  # 먼 거리 임계값
        self.deform_scale = cfg.deform_scale  # 변형 스케일 조정값
        self.attributes_scale = cfg.attributes_scale  # 속성 스케일 조정값
    
    def generate(self, data):
        """
        정점 속성(색상, 변형, 스케일 등)을 생성하는 함수.
        - data: 표정 계수(exp_coeff), 자세(pose), 크기(scale) 등을 포함하는 입력 데이터.
        반환값:
            - 데이터 딕셔너리(업데이트된 xyz, color, scales, rotation 등 포함)
        """
        B = data['exp_coeff'].shape[0]  # 배치 크기

        # 정점 좌표와 특징 복제 (배치 크기만큼)
        xyz = self.xyz.unsqueeze(0).repeat(B, 1, 1)
        feature = torch.tanh(self.feature).unsqueeze(0).repeat(B, 1, 1)

        # 정점과 랜드마크 간 거리 계산
        dists, _, _ = knn_points(xyz, self.landmarks_3d_neutral.unsqueeze(0).repeat(B, 1, 1))
        exp_weights = torch.clamp((self.dist_threshold_far - dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)  # 표정 가중치
        pose_weights = 1 - exp_weights  # 자세 가중치
        exp_controlled = (dists < self.dist_threshold_far).squeeze(-1)  # 표정 영향을 받는 정점
        pose_controlled = (dists > self.dist_threshold_near).squeeze(-1)  # 자세 영향을 받는 정점

        # 초기화: 색상, 변형, 속성
        color = torch.zeros([B, xyz.shape[1], self.exp_color_mlp.dims[-1]], device=xyz.device)
        delta_xyz = torch.zeros_like(xyz, device=xyz.device)
        delta_attributes = torch.zeros([B, xyz.shape[1], self.scales.shape[1] + self.rotation.shape[1] + self.opacity.shape[1]], device=xyz.device)

        # 배치 내 각 샘플 처리 (각 배치에 대해 표정과 자세 기반으로 색상 및 변형을 계산)
        for b in range(B):
            # 표정 기반 색상 생성
            # exp_controlled는 표정에 영향을 받는 정점들의 인덱스를 포함하고 있으며, 이를 통해 해당 정점들의 특성을 추출
            feature_exp_controlled = feature[b, exp_controlled[b], :]  # 표정에 영향을 받는 정점들의 특징 (특성 벡터)
            
            # 표정 정보(`exp_coeff`)와 해당 정점들의 특징(`feature_exp_controlled`)을 결합하여 색상 예측을 위한 입력 생성
            exp_color_input = torch.cat([feature_exp_controlled.t(), data['exp_coeff'][b].unsqueeze(-1).repeat(1, feature_exp_controlled.shape[0])], 0)[None]
            
            # MLP 모델을 사용해 색상 예측
            exp_color = self.exp_color_mlp(exp_color_input)[0].t()  # 예측된 색상 벡터 (MLP를 통해 생성)
            
            # 가중치를 적용하여 표정에 영향을 받는 정점들의 색상을 업데이트
            color[b, exp_controlled[b], :] += exp_color * exp_weights[b, exp_controlled[b], :]  # 가중치(exp_weights)를 적용하여 색상 추가

            # 자세 기반 색상 생성
            # pose_controlled는 자세에 영향을 받는 정점들의 인덱스를 포함하고 있으며, 이를 통해 해당 정점들의 특징을 추출
            feature_pose_controlled = feature[b, pose_controlled[b], :]  # 자세에 영향을 받는 정점들의 특징
            
            # 자세 정보(`pose`)와 해당 정점들의 특징(`feature_pose_controlled`)을 결합하여 색상 예측을 위한 입력 생성
            pose_color_input = torch.cat([feature_pose_controlled.t(), self.pos_embedding(data['pose'][b]).unsqueeze(-1).repeat(1, feature_pose_controlled.shape[0])], 0)[None]
            
            # MLP 모델을 사용해 자세 기반 색상 예측
            pose_color = self.pose_color_mlp(pose_color_input)[0].t()  # 예측된 색상 벡터 (MLP를 통해 생성)
            
            # 가중치를 적용하여 자세에 영향을 받는 정점들의 색상을 업데이트
            color[b, pose_controlled[b], :] += pose_color * pose_weights[b, pose_controlled[b], :]  # 가중치(pose_weights)를 적용하여 색상 추가

            # 표정 및 자세 기반 변형 계산
            # 표정에 영향을 받는 정점들에 대해 변형을 계산
            xyz_exp_controlled = xyz[b, exp_controlled[b], :]  # 표정에 영향을 받는 정점들의 3D 좌표
            
            # 표정 정보(`exp_coeff`)와 해당 정점들의 3D 좌표(`xyz_exp_controlled`)를 결합하여 변형 예측을 위한 입력 생성
            exp_deform_input = torch.cat([self.pos_embedding(xyz_exp_controlled).t(), data['exp_coeff'][b].unsqueeze(-1).repeat(1, xyz_exp_controlled.shape[0])], 0)[None]
            
            # MLP 모델을 사용해 표정에 의한 변형 예측
            exp_deform = self.exp_deform_mlp(exp_deform_input)[0].t()  # 예측된 변형 벡터 (MLP를 통해 생성)
            
            # 가중치를 적용하여 표정에 의한 변형을 해당 정점들의 좌표에 더함
            delta_xyz[b, exp_controlled[b], :] += exp_deform * exp_weights[b, exp_controlled[b], :]  # 표정 변형을 좌표에 추가

            # 자세에 영향을 받는 정점들에 대해 변형을 계산
            xyz_pose_controlled = xyz[b, pose_controlled[b], :]  # 자세에 영향을 받는 정점들의 3D 좌표
            
            # 자세 정보(`pose`)와 해당 정점들의 3D 좌표(`xyz_pose_controlled`)를 결합하여 변형 예측을 위한 입력 생성
            pose_deform_input = torch.cat([self.pos_embedding(xyz_pose_controlled).t(), self.pos_embedding(data['pose'][b]).unsqueeze(-1).repeat(1, xyz_pose_controlled.shape[0])], 0)[None]
            
            # MLP 모델을 사용해 자세에 의한 변형 예측
            pose_deform = self.pose_deform_mlp(pose_deform_input)[0].t()  # 예측된 변형 벡터 (MLP를 통해 생성)
            
            # 가중치를 적용하여 자세에 의한 변형을 해당 정점들의 좌표에 더함
            delta_xyz[b, pose_controlled[b], :] += pose_deform * pose_weights[b, pose_controlled[b], :]  # 자세 변형을 좌표에 추가


        # 정점 좌표 업데이트
        # xyz는 현재 정점들의 3D 좌표입니다. 
        # delta_xyz는 표정 및 자세 변형에 의해 변경된 정점들의 좌표 변화량입니다.
        # 각 변화량에 deform_scale을 곱하여 최종적으로 새로운 3D 좌표를 계산합니다.
        # delta_xyz * self.deform_scale은 변형 정도를 조절하는 역할을 합니다.
        xyz = xyz + delta_xyz * self.deform_scale

        # 스케일, 회전, 불투명도 업데이트
        # delta_attributes는 표정 및 자세에 따른 각 정점의 스케일, 회전, 불투명도 변화량을 포함하고 있습니다.
        # delta_scales는 스케일 변화량을 나타냅니다.
        delta_scales = delta_attributes[:, :, 0:3]

        # 기존의 스케일(self.scales)에 변화량을 더하고, attributes_scale을 곱하여 최종 스케일을 계산합니다.
        # torch.exp()는 스케일 값을 지수 함수로 변환하여 스케일 값을 양수로 만듭니다.
        # 이는 스케일이 음수나 0이 되는 것을 방지하고, 자연스러운 크기 변화를 보장합니다.
        scales = self.scales.unsqueeze(0).repeat(B, 1, 1) + delta_scales * self.attributes_scale
        scales = torch.exp(scales)

        # delta_rotation은 회전 변화량을 나타냅니다.
        # 회전 변화량을 기존의 회전 값에 더하고, attributes_scale을 곱하여 최종 회전 행렬을 계산합니다.
        delta_rotation = delta_attributes[:, :, 3:7]
        rotation = self.rotation.unsqueeze(0).repeat(B, 1, 1) + delta_rotation * self.attributes_scale

        # 회전 행렬은 정규화(normalization)하여 유효한 회전 벡터로 만들어야 합니다.
        # `torch.nn.functional.normalize()`는 벡터의 크기를 1로 정규화하여 올바른 회전 벡터를 유지하게 합니다.
        rotation = torch.nn.functional.normalize(rotation, dim=2)  # 정규화

        # delta_opacity는 불투명도(Opacity) 변화량을 나타냅니다.
        # 기존의 불투명도(self.opacity)에 변화량을 더하고, attributes_scale을 곱하여 최종 불투명도를 계산합니다.
        delta_opacity = delta_attributes[:, :, 7:8]
        opacity = self.opacity.unsqueeze(0).repeat(B, 1, 1) + delta_opacity * self.attributes_scale

        # 불투명도는 sigmoid 함수를 통해 0에서 1 사이의 값으로 변환됩니다.
        # `torch.sigmoid()`는 입력 값을 0~1 범위로 압축하여, 불투명도가 적절한 범위로 조정됩니다.
        opacity = torch.sigmoid(opacity)  # Sigmoid로 0~1 범위로 변환


        # 자세 변환 반영
        if 'pose' in data:
            R = so3_exponential_map(data['pose'][:, :3])  # 자세 회전 행렬
            T = data['pose'][:, None, 3:]  # 자세 이동 벡터
            S = data['scale'][:, :, None]  # 자세 스케일
            xyz = torch.bmm(xyz * S, R.permute(0, 2, 1)) + T  # 좌표 변환

            # 회전 변환 반영
            rotation_matrix = quaternion_to_matrix(rotation)
            rotation_matrix = rearrange(rotation_matrix, 'b n x y -> (b n) x y')
            R = rearrange(R.unsqueeze(1).repeat(1, rotation.shape[1], 1, 1), 'b n x y -> (b n) x y')
            rotation_matrix = rearrange(torch.bmm(R, rotation_matrix), '(b n) x y -> b n x y', b=B)
            rotation = matrix_to_quaternion(rotation_matrix)

            # 스케일 반영
            scales = scales * S

        # 업데이트된 데이터 반환
        data['exp_deform'] = exp_deform
        data['xyz'] = xyz
        data['color'] = color
        data['scales'] = scales
        data['rotation'] = rotation
        data['opacity'] = opacity
        return data
