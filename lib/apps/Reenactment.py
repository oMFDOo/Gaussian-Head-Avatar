import torch
import torch.nn.functional as F
from tqdm import tqdm

class Reenactment():
    """
    Reenactment 클래스:
    - 주어진 데이터를 기반으로 얼굴 애니메이션 생성과 같은 reenactment (재현) 작업을 수행합니다.
    - 이 클래스는 GaussianHead 모듈을 사용하여 얼굴의 모습을 생성하고, 
      초고해상도 이미지를 생성하는 SuperResolution 모듈을 사용하여 품질을 향상시킵니다.
    """
    
    def __init__(self, dataloader, gaussianhead, supres, camera, recorder, gpu_id, freeview):
        """
        초기화 함수:
        - 클래스의 인스턴스를 초기화합니다. 
        - 데이터 로더(dataloader), GaussianHead 모델, SuperResolution 모델, 카메라 모듈,
          기록(recorder) 모듈, GPU ID, 그리고 freeview 모드 설정을 받아옵니다.

        Args:
            dataloader: 훈련 데이터를 제공하는 데이터로더
            gaussianhead: 얼굴을 생성하는 모델
            supres: 초고해상도 이미지를 생성하는 모델
            camera: 카메라 모듈
            recorder: 학습 로그와 결과를 기록하는 모듈
            gpu_id: 사용할 GPU ID
            freeview: 자유로운 카메라 뷰 설정 여부
        """
        self.dataloader = dataloader
        self.gaussianhead = gaussianhead  # 얼굴 생성 모델
        self.supres = supres  # 초고해상도 모델
        self.camera = camera  # 카메라 모듈
        self.recorder = recorder  # 기록 모듈
        self.device = torch.device('cuda:%d' % gpu_id)  # GPU 설정
        self.freeview = freeview  # 자유로운 카메라 뷰 여부

    def run(self):
        """
        run 함수:
        - 데이터셋을 반복하면서 얼굴 생성과 초고해상도 이미지를 순차적으로 처리합니다.
        - 데이터는 `dataloader`에서 가져오며, 각 배치에 대해 처리하고, 기록을 남깁니다.
        """
        # 데이터로더를 순차적으로 가져옴 (tqdm을 사용하여 진행 상황 표시)
        for idx, data in tqdm(enumerate(self.dataloader)):

            # 데이터 항목들을 모두 GPU로 전송 (텐서로 처리하기 위해)
            to_cuda = ['images', 'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 
                       'full_proj_transform', 'camera_center', 'pose', 'scale', 'exp_coeff', 'pose_code']
            for data_item in to_cuda:
                data[data_item] = data[data_item].to(device=self.device)

            # freeview 모드가 아닐 때, 이전 값들을 혼합하여 부드러운 변화를 만듦
            if not self.freeview:
                if idx > 0:  # 첫 번째 배치 이후부터 실행
                    # 이전 pose와 exp_coeff를 혼합하여 더 부드러운 변화 유도
                    data['pose'] = pose_last * 0.5 + data['pose'] * 0.5
                    data['exp_coeff'] = exp_last * 0.5 + data['exp_coeff'] * 0.5
                pose_last = data['pose']  # 현재 pose를 저장
                exp_last = data['exp_coeff']  # 현재 exp_coeff를 저장
                
            else:
                # freeview 모드일 경우, pose를 0으로 초기화하여 모든 카메라에서 동일한 모습 생성
                data['pose'] *= 0
                if idx > 0:  # 첫 번째 배치 이후부터 실행
                    data['exp_coeff'] = exp_last * 0.5 + data['exp_coeff'] * 0.5
                exp_last = data['exp_coeff']  # 현재 exp_coeff를 저장
            
            # `torch.no_grad()`를 사용하여 모델을 평가 모드로 설정 (역전파 계산을 하지 않음)
            with torch.no_grad():
                # 얼굴 생성 모델을 사용하여 데이터를 처리
                data = self.gaussianhead.generate(data)
                # 카메라 모듈을 사용하여 얼굴 이미지를 렌더링 (해상도 512로 생성)
                data = self.camera.render_gaussian(data, 512)
                render_images = data['render_images']  # 렌더링된 이미지

                # 초고해상도 이미지를 생성하는 모델을 사용하여 품질 향상
                supres_images = self.supres(render_images)  # 초고해상도 이미지 생성
                data['supres_images'] = supres_images  # 초고해상도 이미지를 데이터에 저장

            # 학습 로그에 기록할 데이터 준비
            log = {
                'data': data,  # 처리된 데이터
                'iter': idx  # 현재 반복 횟수
            }

            # 기록 모듈을 사용하여 학습 상태 저장
            self.recorder.log(log)  # 학습 로그 저장
