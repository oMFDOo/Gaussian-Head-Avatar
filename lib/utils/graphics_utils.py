#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

# NamedTuple을 사용하여 기본 포인트 클라우드 구조 정의
class BasicPointCloud(NamedTuple):
    points: np.array  # 점들의 3D 좌표 (Nx3)
    colors: np.array  # 각 점의 색상 정보 (Nx3)
    normals: np.array  # 각 점의 법선 벡터 (Nx3)

def geom_transform_points(points, transf_matrix):
    """
    점들을 주어진 변환 행렬에 따라 변환.
    입력:
        points - Nx3 크기의 점 좌표
        transf_matrix - 4x4 크기의 변환 행렬 (평행 이동 및 회전 포함)
    출력:
        변환된 점 좌표 (Nx3)
    """
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)  # 동차 좌표(homogeneous coordinate) 추가
    points_hom = torch.cat([points, ones], dim=1)  # Nx4 형태로 확장
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))  # 변환 행렬 적용

    denom = points_out[..., 3:] + 1e-7  # 동차 좌표의 마지막 값으로 나누기 (0으로 나누는 것을 방지)
    return (points_out[..., :3] / denom).squeeze(dim=0)  # Nx3 형태로 변환 후 반환

def getWorld2View(R, t):
    """
    월드 좌표계를 뷰(카메라) 좌표계로 변환하는 행렬 생성.
    입력:
        R - 3x3 회전 행렬
        t - 3x1 평행 이동 벡터
    출력:
        4x4 변환 행렬
    """
    Rt = np.zeros((4, 4))  # 4x4 행렬 초기화
    Rt[:3, :3] = R.transpose()  # 회전 행렬의 전치
    Rt[:3, 3] = t  # 평행 이동 벡터 추가
    Rt[3, 3] = 1.0  # 동차 좌표 형식 유지
    return np.float32(Rt)  # 32비트 float로 반환

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    월드 좌표계를 뷰 좌표계로 변환하면서 추가적인 평행 이동 및 스케일링 적용.
    입력:
        R - 3x3 회전 행렬
        t - 3x1 평행 이동 벡터
        translate - 추가적인 평행 이동
        scale - 스케일링 값
    출력:
        4x4 변환 행렬
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)  # 카메라에서 월드 좌표계로 변환
    cam_center = C2W[:3, 3]  # 카메라 중심 추출
    cam_center = (cam_center + translate) * scale  # 추가 변환 적용
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)  # 다시 월드에서 카메라 좌표계로 변환
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    프로젝션(투영) 행렬 생성.
    입력:
        znear - 카메라 근평면
        zfar - 카메라 원평면
        fovX - 수평 시야각 (라디안 단위)
        fovY - 수직 시야각 (라디안 단위)
    출력:
        4x4 프로젝션 행렬
    """
    # 시야각(FOV)을 기반으로 근평면에서의 절두체(frustum) 경계 계산
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)  # 4x4 행렬 초기화

    z_sign = 1.0  # OpenGL 스타일 프로젝션 사용

    # 프로젝션 행렬 값 설정
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    """
    시야각(FOV)을 초점 거리(focal length)로 변환.
    입력:
        fov - 시야각 (라디안 단위)
        pixels - 이미지 해상도 (픽셀 단위)
    출력:
        초점 거리 (focal length)
    """
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    """
    초점 거리(focal length)를 시야각(FOV)으로 변환.
    입력:
        focal - 초점 거리
        pixels - 이미지 해상도 (픽셀 단위)
    출력:
        시야각 (라디안 단위)
    """
    return 2 * math.atan(pixels / (2 * focal))
