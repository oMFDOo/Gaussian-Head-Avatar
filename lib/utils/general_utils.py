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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    """
    Sigmoid 함수의 역변환을 계산.
    입력: x (Tensor) - sigmoid 출력값
    출력: sigmoid 입력값을 복원한 값
    """
    return torch.log(x / (1 - x))

def PILtoTorch(pil_image, resolution):
    """
    PIL 이미지를 PyTorch 텐서로 변환.
    입력:
        pil_image - PIL 이미지 객체
        resolution - 원하는 출력 해상도 (가로, 세로)
    출력:
        PyTorch 텐서로 변환된 이미지
    """
    resized_image_PIL = pil_image.resize(resolution)  # PIL 이미지 크기 조정
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0  # 0~255 범위를 0~1로 정규화
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)  # (H, W, C)를 (C, H, W)로 변환
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)  # 단일 채널 이미지를 (C, H, W)로 변환

def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    
    학습률 스케줄러를 생성. 초기값과 최종값을 주어진 단계 수에 따라 지수적으로 감소.
    입력:
        lr_init - 초기 학습률
        lr_final - 최종 학습률
        lr_delay_steps - 지연 단계 수
        lr_delay_mult - 지연 시 학습률 비율
        max_steps - 전체 단계 수
    출력:
        스케줄러 함수
    """
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0  # 학습률이 없을 경우 0 반환
        if lr_delay_steps > 0:
            # 학습 초기 지연 단계에서 학습률 완화
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)  # 현재 단계 비율
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)  # 로그 선형 보간
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    """
    대각 하부 요소를 추출하여 불확실성 계산.
    입력:
        L - 하부 삼각 행렬 (Nx3x3)
    출력:
        불확실성 (Nx6)
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    # 대각 요소와 오프 대각 요소 추출
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    """
    대칭 행렬에서 불확실성 추출 (strip_lowerdiag와 동일 기능).
    """
    return strip_lowerdiag(sym)

def build_rotation(r):
    """
    쿼터니언을 기반으로 회전 행렬 생성.
    입력:
        r - 쿼터니언 (Nx4)
    출력:
        회전 행렬 (Nx3x3)
    """
    norm = torch.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2 + r[:, 3]**2)  # 쿼터니언 크기
    q = r / norm[:, None]  # 정규화

    # 각 성분
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 회전 행렬 계산
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def build_scaling_rotation(s, r):
    """
    스케일링 및 회전을 결합한 행렬 생성.
    입력:
        s - 스케일링 요소 (Nx3)
        r - 회전 쿼터니언 (Nx4)
    출력:
        스케일링 및 회전 행렬 (Nx3x3)
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=r.device)
    R = build_rotation(r)  # 회전 행렬 생성

    # 스케일링 적용
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L  # 회전과 결합
    return L

def Rotate_y_180(X, pos='right'):
    """
    Y축 기준 180도 회전.
    입력:
        X - 입력 행렬
        pos - 'right' 또는 'left' (회전 방향)
    출력:
        회전된 행렬
    """
    R = torch.eye(3).to(X.device)
    R[0, 0] = -1.0
    R[2, 2] = -1.0
    if pos == 'right':
        X = torch.matmul(X, R)  # 오른쪽 곱셈
    else:
        X = torch.matmul(R, X)  # 왼쪽 곱셈
    return X

def Rotate_z_180(X, pos='right'):
    """
    Z축 기준 180도 회전.
    입력:
        X - 입력 행렬
        pos - 'right' 또는 'left' (회전 방향)
    출력:
        회전된 행렬
    """
    R = torch.eye(3).to(X.device)
    R[0, 0] = -1.0
    R[1, 1] = -1.0
    if pos == 'right':
        X = torch.matmul(X, R)  # 오른쪽 곱셈
    else:
        X = torch.matmul(R, X)  # 왼쪽 곱셈
    return X
