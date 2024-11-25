import torch
from prefetch_generator import BackgroundGenerator

class DataLoaderX(torch.utils.data.DataLoader):
    """
    DataLoaderX 클래스:
    - 기본 DataLoader에 백그라운드에서 데이터를 비동기적으로 로드하는 기능을 추가한 클래스입니다.
    - 데이터를 효율적으로 불러오기 위해 `BackgroundGenerator`를 활용하여 데이터 로딩 속도를 향상시킵니다.
    """

    def __iter__(self):
        """
        __iter__ 함수:
        - 데이터 로딩을 비동기적으로 수행하여 GPU/CPU가 데이터를 처리하는 동안 데이터를 미리 로드하도록 합니다.
        - super().__iter__()는 기본 DataLoader의 데이터를 반복적으로 가져오는 메서드입니다.
        - `BackgroundGenerator`를 사용해 데이터를 백그라운드에서 미리 로드하고, 메인 프로세스에서 데이터를 바로 사용할 수 있도록 합니다.

        반환값:
        - `BackgroundGenerator`를 사용하여 데이터를 비동기적으로 가져오는 반복자(iterator)를 반환합니다.
        """
        return BackgroundGenerator(super().__iter__())  # 기본 DataLoader의 __iter__() 호출 후 BackgroundGenerator로 감싸서 반환
