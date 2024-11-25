import torch
import torch.nn as nn

# 포지셔널 인코딩 (Positional Encoding) 클래스
# 이 클래스는 입력 데이터를 주파수 기반으로 변환하여 고주파 정보를 학습 가능하게 만듦 (NeRF의 section 5.1 참고)
class Embedder:
    def __init__(self, **kwargs):
        """
        Embedder 초기화 함수.
        - kwargs: 다음 매개변수를 포함하는 딕셔너리
            - include_input: 입력 데이터를 결과에 포함할지 여부
            - input_dims: 입력 차원
            - max_freq_log2: 최대 주파수의 로그2 값
            - num_freqs: 사용할 주파수 밴드 개수
            - log_sampling: 로그 스케일로 주파수 밴드를 생성할지 여부
            - periodic_fns: 사용할 주기 함수 리스트 (예: [torch.sin, torch.cos])
        """
        self.kwargs = kwargs
        self.create_embedding_fn()  # 인코딩 함수 생성

    def create_embedding_fn(self):
        """
        인코딩 함수 생성.
        - 주어진 설정에 따라 입력 데이터를 변환할 함수를 생성.
        """
        embed_fns = []  # 인코딩 함수를 저장할 리스트
        d = self.kwargs['input_dims']  # 입력 차원
        out_dim = 0  # 출력 차원 초기화

        # 입력 데이터를 결과에 포함할 경우
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)  # 입력 그대로 추가
            out_dim += d  # 출력 차원 증가

        # 주파수 밴드 생성
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            # 로그 스케일로 주파수 밴드 생성
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            # 선형 스케일로 주파수 밴드 생성
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        # 주기 함수 (sin, cos 등)와 주파수 밴드를 조합하여 인코딩 함수 생성
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))  # 주기 함수와 주파수 적용
                out_dim += d  # 출력 차원 증가

        self.embed_fns = embed_fns  # 생성된 인코딩 함수 저장
        self.out_dim = out_dim  # 최종 출력 차원 저장

    def embed(self, inputs):
        """
        입력 데이터를 인코딩.
        - inputs: 인코딩할 입력 데이터 (Tensor)
        반환값:
            인코딩된 텐서.
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)  # 모든 인코딩 함수의 결과를 결합


def get_embedder(multires, i=0):
    """
    Embedder 객체를 생성하고, 인코딩 함수와 출력 차원을 반환.
    - multires: 다중 해상도를 위한 주파수 개수
    - i: -1일 경우 Identity 함수 반환, 그렇지 않으면 Embedder 사용
    반환값:
        - 인코딩 함수
        - 인코딩 출력 차원
    """
    if i == -1:
        # 인코딩을 사용하지 않을 경우, 입력 그대로 반환하는 Identity 함수 반환
        return nn.Identity(), 3

    # Embedder 설정 생성
    embed_kwargs = {
        'include_input': True,  # 입력 데이터를 포함
        'input_dims': 1,  # 입력 차원 (예: 위치 좌표)
        'max_freq_log2': multires-1,  # 최대 주파수의 로그2 값
        'num_freqs': multires,  # 사용할 주파수 밴드 개수
        'log_sampling': True,  # 로그 스케일로 주파수 밴드 생성
        'periodic_fns': [torch.sin, torch.cos],  # 사용할 주기 함수
    }

    # Embedder 객체 생성
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)  # 인코딩 함수 생성
    return embed, embedder_obj.out_dim  # 인코딩 함수와 출력 차원 반환
