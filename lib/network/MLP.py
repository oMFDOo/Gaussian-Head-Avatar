import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    다층 퍼셉트론 (MLP) 클래스.
    - 입력 데이터를 다중 비선형 변환을 통해 학습하는 구조.
    """
    def __init__(self, dims, last_op=None):
        """
        MLP 초기화 함수.
        - dims: 각 층의 뉴런 개수를 정의하는 리스트. 예: [64, 128, 64, 3].
        - last_op: 최종 출력에 적용할 함수 (예: 활성화 함수).
        """
        super(MLP, self).__init__()

        self.dims = dims
        self.skip_layer = [int(len(dims) / 2)]  # 중간 층에서 skip 연결을 추가할 레이어 인덱스
        self.last_op = last_op  # 마지막 층에서 적용할 연산

        self.layers = []  # 레이어를 담을 리스트
        for l in range(0, len(dims) - 1):
            if l in self.skip_layer:
                # skip_layer에서는 첫 번째 입력과 현재 출력 채널을 결합하여 처리
                self.layers.append(nn.Conv1d(dims[l] + dims[0], dims[l + 1], 1))
            else:
                # 일반적인 1D 컨볼루션 레이어
                self.layers.append(nn.Conv1d(dims[l], dims[l + 1], 1))
            self.add_module("conv%d" % l, self.layers[l])  # 레이어를 모듈로 등록

    def forward(self, latet_code, return_all=False):
        """
        MLP의 순전파 함수.
        - latet_code: 입력 텐서.
        - return_all: True일 경우 모든 레이어의 출력을 반환.
        반환값:
            최종 출력 또는 모든 레이어의 출력.
        """
        y = latet_code  # 입력 데이터
        tmpy = latet_code  # skip 연결을 위한 초기 데이터 저장
        y_list = []  # 모든 레이어 출력을 저장할 리스트

        for l, f in enumerate(self.layers):
            if l in self.skip_layer:
                # skip_layer에서는 첫 입력과 현재 출력을 결합
                y = self._modules['conv' + str(l)](torch.cat([y, tmpy], 1))
            else:
                # 일반적인 레이어 처리
                y = self._modules['conv' + str(l)](y)
            if l != len(self.layers) - 1:
                # 마지막 레이어가 아닌 경우 활성화 함수 적용 (Leaky ReLU)
                y = F.leaky_relu(y)
        if self.last_op:
            # 마지막 연산이 정의되어 있다면 적용
            y = self.last_op(y)
            y_list.append(y)
        if return_all:
            # 모든 레이어 출력을 반환
            return y_list
        else:
            # 최종 출력만 반환
            return y
