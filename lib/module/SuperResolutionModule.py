import torch
from torch import nn
from einops import rearrange

from lib.network.Upsampler import Upsampler

class SuperResolutionModule(nn.Module):
    """
    SuperResolutionModule:
    - 입력 이미지를 고해상도로 변환하는 모듈입니다.
    - 내부적으로 'Upsampler' 네트워크를 사용하여 입력 이미지를 업샘플링합니다.
    """

    def __init__(self, cfg):
        """
        초기화 함수:
        - cfg: 네트워크 구성(configuration) 정보, 네트워크 파라미터들 (입력/출력 차원, 네트워크 용량 등)을 포함합니다.
        
        주요 역할:
        1. Upsampler 네트워크 초기화
        2. 입력 및 출력 차원, 네트워크 용량을 기반으로 Upsampler 네트워크를 설정합니다.
        """
        super(SuperResolutionModule, self).__init__()

        # Upsampler 네트워크 초기화: 입력 차원, 출력 차원, 네트워크 용량 등을 설정
        self.upsampler = Upsampler(cfg.input_dim, cfg.output_dim, cfg.network_capacity)

    def forward(self, input):
        """
        순전파 함수:
        - 입력된 데이터를 고해상도로 변환합니다.
        
        입력:
        - input: 저해상도 이미지 텐서 (예: (B, C, H, W) 형태, B: 배치 크기, C: 채널, H: 높이, W: 너비)

        출력:
        - output: 업샘플링된 고해상도 이미지 텐서
        """
        # Upsampler 네트워크를 사용하여 입력을 업샘플링
        output = self.upsampler(input)
        
        return output
