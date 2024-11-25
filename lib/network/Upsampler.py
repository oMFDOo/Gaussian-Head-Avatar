import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    두 개의 연속적인 Conv2D 레이어를 적용하는 모듈.
    각 Conv2D는 InstanceNorm과 ReLU 활성화 함수가 뒤따릅니다.
    구조: (Conv2D => [InstanceNorm] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        초기화 함수.
        - in_channels: 입력 채널 수
        - out_channels: 출력 채널 수
        - mid_channels: 중간 채널 수 (생략 시 out_channels로 설정)
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # 첫 번째 Conv2D
            nn.InstanceNorm2d(mid_channels),  # 첫 번째 InstanceNorm
            nn.ReLU(inplace=True),  # 첫 번째 ReLU
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),  # 두 번째 Conv2D
            nn.InstanceNorm2d(out_channels),  # 두 번째 InstanceNorm
            nn.ReLU(inplace=True)  # 두 번째 ReLU
        )

    def forward(self, x):
        """
        순전파 함수.
        - x: 입력 텐서
        반환값: DoubleConv 모듈을 통과한 출력 텐서
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    다운샘플링 모듈.
    MaxPool2D로 해상도를 절반으로 줄인 후 DoubleConv 적용.
    """
    def __init__(self, in_channels, out_channels):
        """
        초기화 함수.
        - in_channels: 입력 채널 수
        - out_channels: 출력 채널 수
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2 영역에서 Max Pooling 수행
            DoubleConv(in_channels, out_channels)  # DoubleConv 적용
        )

    def forward(self, x):
        """
        순전파 함수.
        - x: 입력 텐서
        반환값: 다운샘플링 및 DoubleConv 결과 텐서
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    업샘플링 모듈.
    Bilinear Upsampling을 수행한 후 DoubleConv 적용.
    """
    def __init__(self, in_channels, out_channels):
        """
        초기화 함수.
        - in_channels: 입력 채널 수
        - out_channels: 출력 채널 수
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 업샘플링
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)  # DoubleConv

    def forward(self, x1, x2=None):
        """
        순전파 함수.
        - x1: 업샘플링할 입력 텐서
        - x2: 스킵 연결 텐서 (옵션)
        반환값: 업샘플링 및 DoubleConv 결과 텐서
        """
        x1 = self.up(x1)  # 업샘플링 수행
        if x2 is not None:
            # x2의 크기와 맞추기 위해 패딩 추가
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)  # 스킵 연결 (채널 방향으로 결합)
        else:
            x = x1  # 스킵 연결 없이 진행
        return self.conv(x)  # DoubleConv 적용


class OutConv(nn.Module):
    """
    출력 컨볼루션 모듈.
    1x1 Conv2D로 채널 수를 원하는 출력 채널로 변환.
    Sigmoid 활성화 함수로 출력 값을 0~1로 정규화.
    """
    def __init__(self, in_channels, out_channels):
        """
        초기화 함수.
        - in_channels: 입력 채널 수
        - out_channels: 출력 채널 수
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 Conv2D
        self.act_fn = nn.Sigmoid()  # Sigmoid 활성화 함수

    def forward(self, x):
        """
        순전파 함수.
        - x: 입력 텐서
        반환값: 1x1 Conv2D와 Sigmoid를 거친 출력 텐서
        """
        y = self.act_fn(self.conv(x))
        return y


class Upsampler(nn.Module):
    """
    업샘플링 네트워크.
    입력 텐서를 점진적으로 업샘플링하며 원하는 해상도의 이미지를 생성.
    """
    def __init__(self, input_dim=32, output_dim=3, network_capacity=128):
        """
        초기화 함수.
        - input_dim: 입력 채널 수
        - output_dim: 출력 채널 수
        - network_capacity: 네트워크 용량 (채널 크기 조정)
        """
        super(Upsampler, self).__init__()
        self.inc = DoubleConv(input_dim, network_capacity * 4)  # 초기 DoubleConv
        self.up1 = Up(network_capacity * 4, network_capacity * 2)  # 첫 번째 업샘플링
        self.up2 = Up(network_capacity * 2, network_capacity)  # 두 번째 업샘플링
        self.outc = OutConv(network_capacity, output_dim)  # 출력 Conv

    def forward(self, x):
        """
        순전파 함수.
        - x: 입력 텐서
        반환값: 업샘플링된 최종 출력
        """
        x = self.inc(x)  # 초기 DoubleConv
        x = self.up1(x)  # 첫 번째 업샘플링
        x = self.up2(x)  # 두 번째 업샘플링
        x = self.outc(x)  # 출력 Conv
        return x
