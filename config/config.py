import os
from yacs.config import CfgNode as CN
 

class config_base():

    def __init__(self):
        self.cfg = CN()
    
    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self,config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()


class config_train(config_base):

    def __init__(self):
        super(config_train, self).__init__()

        self.cfg.gpu_id = 0                                     # 사용할 GPU ID : 모델 학습 시 사용하는 GPU 장치를 선택합니다. 여러 GPU를 사용할 경우 GPU ID를 지정합니다.
        self.cfg.load_meshhead_checkpoint = ''                  # 메쉬 헤드의 체크포인트 경로 : 학습이 중단된 경우, 이전 학습의 가중치를 복원하여 이어서 학습하기 위해 사용하는 파일 경로입니다.
        self.cfg.load_gaussianhead_checkpoint = ''              # 가우시안 헤드의 체크포인트 경로 : Gaussian Head 모듈의 사전 학습된 가중치가 저장된 파일 경로입니다.
        self.cfg.load_supres_checkpoint = ''                    # 초해상도 네트워크의 체크포인트 경로 : Super-Resolution 네트워크의 사전 학습된 가중치를 로드하기 위한 경로입니다.
        self.cfg.load_delta_poses_checkpoint = ''               # 머리 자세의 프레임별 오프셋 체크포인트 경로 : 프레임별 머리 자세 보정을 위한 delta_poses 네트워크의 체크포인트 경로입니다.
        self.cfg.lr_net = 0.0                                   # 모델 및 네트워크 학습률 : 주요 네트워크(예: MeshHead, GaussianHead 등)의 가중치를 업데이트하는 데 사용되는 학습률 값입니다.
        self.cfg.lr_lmk = 0.0                                   # 3D 랜드마크 학습률 : 3D 얼굴 랜드마크 좌표를 최적화하는 데 사용되는 학습률 값입니다.
        self.cfg.lr_pose = 0.0                                  # delta_poses 학습률 : 머리 자세의 오프셋(보정 값)을 최적화하는 데 사용되는 학습률 값입니다.
        self.cfg.batch_size = 1                                 # 권장 배치 크기 = 1 : 학습 중 한 번의 반복(iteration) 동안 네트워크에 공급할 데이터 샘플의 개수를 설정합니다. GPU 메모리 제한으로 인해 일반적으로 1로 설정됩니다.
        self.cfg.optimize_pose = False                          # delta_poses 최적화 여부 : True로 설정하면 delta_poses를 학습하며, False로 설정하면 고정된 상태로 유지됩니다.

        self.cfg.dataset = CN()
        self.cfg.dataset.dataroot = ''                          # 데이터셋의 루트 디렉토리 : 데이터셋(이미지 및 파라미터 파일 등)이 저장된 최상위 디렉토리 경로를 지정합니다.
        self.cfg.dataset.camera_ids = []                        # 사용할 카메라 ID : 여러 카메라 뷰를 사용하는 경우 학습에 포함할 특정 카메라 ID를 나열합니다.
        self.cfg.dataset.original_resolution = 2048             # 원본 이미지 해상도 : 데이터셋의 원본 이미지 해상도로, 카메라의 내재 파라미터와 일치해야 합니다.
        self.cfg.dataset.resolution = 2048                       # 데이터셋 렌더링 해상도 : 학습 및 평가를 위해 원본 이미지를 리샘플링할 해상도입니다. 512는 실시간 처리와 품질 간의 균형을 제공합니다.
        self.cfg.dataset.num_sample_view = 8                    # 메쉬 헤드 학습 중 다양한 뷰에서 샘플링한 이미지 수 : 학습 중 다양한 시점에서 샘플링할 이미지의 수를 지정합니다. 이 값은 모델이 다양한 각도에서 일반화되도록 도와줍니다.

        self.cfg.meshheadmodule = CN()
        self.cfg.meshheadmodule.geo_mlp = []                    # 지오메트리 MLP의 차원 : 3D 메쉬를 나타내는 지오메트리 정보를 학습하기 위해 사용하는 다층 퍼셉트론(MLP)의 레이어 차원 리스트입니다.
        self.cfg.meshheadmodule.exp_color_mlp = []              # 표정 색상 MLP의 차원 : 표정에 따른 색상 변화(예: 피부색 변형)를 모델링하기 위한 MLP의 차원 리스트입니다.
        self.cfg.meshheadmodule.pose_color_mlp = []             # 자세 색상 MLP의 차원 : 얼굴 자세 변화에 따른 색상 변화를 모델링하기 위한 MLP의 차원 리스트입니다.
        self.cfg.meshheadmodule.exp_deform_mlp = []             # 표정 변형 MLP의 차원 : 표정 변화에 따라 메쉬가 변형되는 양상을 모델링하기 위한 MLP의 차원 리스트입니다.
        self.cfg.meshheadmodule.pose_deform_mlp = []            # 자세 변형 MLP의 차원 : 머리 자세에 따른 메쉬 변형을 모델링하기 위한 MLP의 차원 리스트입니다.
        self.cfg.meshheadmodule.pos_freq = 4                    # 위치 인코딩 빈도 : 모델 입력에 대한 주파수 기반 위치 인코딩(Positional Encoding)의 빈도를 설정합니다. 높은 빈도는 더 많은 세부 정보를 학습할 수 있게 합니다.
        self.cfg.meshheadmodule.model_bbox = []                 # 머리 모델의 경계 상자 : 3D 메쉬의 범위를 정의하는 최소 및 최대 좌표값입니다.
        self.cfg.meshheadmodule.dist_threshold_near = 0.1       # 임계값 t1 : 3D 가우시안과 랜드마크의 최소 거리 가중치. 랜드마크가 너무 가까운 경우를 처리합니다.
        self.cfg.meshheadmodule.dist_threshold_far = 0.2        # 임계값 t2 : 3D 가우시안과 랜드마크의 최대 거리 가중치. 랜드마크가 너무 먼 경우를 처리합니다.
        self.cfg.meshheadmodule.deform_scale = 0.3              # 변형 스케일 팩터 : 메쉬 변형 정도를 제어하는 스케일 값입니다. 값이 크면 변형 정도가 증가합니다.
        self.cfg.meshheadmodule.subdivide = False               # 테트메쉬 세분화 여부 : True로 설정하면 메쉬의 해상도를 128에서 256으로 세분화하여 더 정밀한 모델링을 수행합니다.

        self.cfg.supresmodule = CN()
        self.cfg.supresmodule.input_dim = 32                    # 입력 차원 : Super-Resolution 네트워크 입력의 채널 수를 나타냅니다. 멀티채널 색상을 사용합니다.
        self.cfg.supresmodule.output_dim = 3                    # 출력 차원 : Super-Resolution 네트워크의 최종 출력 이미지 채널 수를 나타냅니다(RGB 기준으로 3).
        self.cfg.supresmodule.network_capacity = 64             # 네트워크의 마지막 컨볼루션 계층의 차원 : 네트워크의 복잡도를 조절하며, 값이 클수록 모델 용량과 표현력이 증가합니다.

        self.cfg.gaussianheadmodule = CN()
        self.cfg.gaussianheadmodule.num_add_mouth_points = 0    # 초기화 중 입 랜드마크 주변에 추가되는 점의 수 : 초기화 단계에서 입 주변의 추가 점을 포함하여 모델링을 더 정밀하게 합니다.
        self.cfg.gaussianheadmodule.exp_color_mlp = []          # 표정 색상 MLP의 차원 : Gaussian Head 모듈에서 표정에 따른 색상 변화를 모델링하는 MLP의 레이어 차원 리스트입니다.
        self.cfg.gaussianheadmodule.pose_color_mlp = []         # 자세 색상 MLP의 차원 : Gaussian Head 모듈에서 자세에 따른 색상 변화를 모델링하는 MLP의 레이어 차원 리스트입니다.
        self.cfg.gaussianheadmodule.exp_attributes_mlp = []     # 표정 속성 MLP의 차원 : 표정 속성(attribute) 정보를 모델링하기 위한 MLP의 레이어 차원 리스트입니다.
        self.cfg.gaussianheadmodule.pose_attributes_mlp = []    # 자세 속성 MLP의 차원 : 자세 속성(attribute) 정보를 모델링하기 위한 MLP의 레이어 차원 리스트입니다.
        self.cfg.gaussianheadmodule.exp_deform_mlp = []         # 표정 변형 MLP의 차원 : 표정 변화에 따른 변형을 모델링하기 위한 MLP의 레이어 차원 리스트입니다.
        self.cfg.gaussianheadmodule.pose_deform_mlp = []        # 자세 변형 MLP의 차원 : 자세 변화에 따른 변형을 모델링하기 위한 MLP의 레이어 차원 리스트입니다.
        self.cfg.gaussianheadmodule.exp_coeffs_dim = 64         # 표정 계수의 차원 : 표정 정보를 표현하는 벡터의 차원을 정의합니다.
        self.cfg.gaussianheadmodule.pos_freq = 4                # 위치 인코딩 빈도 : Gaussian Head 모듈에서 주파수 기반 위치 인코딩 빈도를 설정합니다.
        self.cfg.gaussianheadmodule.dist_threshold_near = 0.1   # 임계값 t1 : Gaussian Head에서 3D 가우시안과 랜드마크의 최소 거리 가중치.
        self.cfg.gaussianheadmodule.dist_threshold_far = 0.2    # 임계값 t2 : Gaussian Head에서 3D 가우시안과 랜드마크의 최대 거리 가중치.
        self.cfg.gaussianheadmodule.deform_scale = 0.3          # 변형 스케일 팩터 : Gaussian Head에서 변형 정도를 제어하는 값.
        self.cfg.gaussianheadmodule.attributes_scale = 0.05     # 속성 오프셋의 스케일 팩터 : 속성 값의 크기를 조정하는 데 사용됩니다.

        self.cfg.recorder = CN()
        self.cfg.recorder.name = ''                             # 아바타의 이름 : 결과 데이터와 모델을 식별하기 위한 이름입니다.
        self.cfg.recorder.logdir = ''                           # TensorBoard 로그 디렉토리 : 학습 로그와 시각화 데이터를 저장할 디렉토리입니다.
        self.cfg.recorder.checkpoint_path = ''                  # 체크포인트 저장 경로 : 학습 진행 상황을 저장하기 위한 경로입니다.
        self.cfg.recorder.result_path = ''                      # 시각화 결과 저장 경로 : 학습 후 시각화된 결과 데이터를 저장하는 경로입니다.
        self.cfg.recorder.save_freq = 1                         # 체크포인트 저장 빈도 : 몇 번의 반복(iteration)마다 체크포인트를 저장할지 결정합니다.
        self.cfg.recorder.show_freq = 1                         # 시각화 결과 저장 빈도 : 시각화 결과를 몇 번의 반복마다 저장할지 결정합니다.

class config_reenactment(config_base):

    def __init__(self):
        super(config_reenactment, self).__init__()

        self.cfg.gpu_id = 0                                     # 사용할 GPU ID : 재현 작업 시 사용할 GPU 장치를 선택합니다. GPU 메모리가 충분한지 확인 후 적절한 ID를 설정합니다.
        self.cfg.load_gaussianhead_checkpoint = ''              # 가우시안 헤드의 체크포인트 경로 : Gaussian Head 모듈의 사전 학습된 가중치를 로드하기 위한 경로입니다.
        self.cfg.load_supres_checkpoint = ''                    # 초해상도 네트워크의 체크포인트 경로 : Super-Resolution 네트워크의 사전 학습된 가중치를 로드하기 위한 경로입니다.

        self.cfg.dataset = CN()
        self.cfg.dataset.dataroot = ''                          # 데이터셋의 루트 디렉토리 : 입력 이미지 및 관련 파라미터 파일이 저장된 최상위 디렉토리를 지정합니다.
        self.cfg.dataset.image_files = ''                       # 입력 이미지 파일명 : 재현 작업에서 처리할 입력 이미지의 파일 이름을 지정합니다. 경로는 `dataroot`를 기준으로 상대 경로일 수 있습니다.
        self.cfg.dataset.param_files = ''                       # BFM 파라미터 파일명 : BFM(3D Morphable Model) 파라미터 파일 이름을 지정합니다. 이 파일에는 머리 자세와 표정 계수가 포함됩니다.
        self.cfg.dataset.camera_path = ''                       # 특정 카메라 경로 : 특정 카메라 뷰의 파라미터를 포함하는 파일 경로입니다. 자유 시점 렌더링이 아닐 경우 사용됩니다.
        self.cfg.dataset.pose_code_path = ''                    # 특정 포즈 코드 경로 : 네트워크 입력으로 사용되는 특정 자세 코드를 지정하는 파일 경로입니다.
        self.cfg.dataset.freeview = False                       # 자유 시점 렌더링 여부 : True로 설정하면 고정된 카메라 뷰 대신 자유 시점으로 렌더링합니다.
        self.cfg.dataset.original_resolution = 2048             # 원본 이미지 해상도 : 데이터셋의 원본 이미지 해상도로, 내재적 파라미터와 일치해야 합니다.
        self.cfg.dataset.resolution = 2048                       # 렌더링 해상도 : 재현 작업에서 렌더링된 이미지를 저장할 해상도를 지정합니다. 낮은 값은 더 빠른 처리 속도를 제공합니다.

        self.cfg.supresmodule = CN()
        self.cfg.supresmodule.input_dim = 32                    # 입력 차원 : Super-Resolution 네트워크 입력의 채널 수로, 다채널 색상 데이터를 처리합니다.
        self.cfg.supresmodule.output_dim = 3                    # 출력 차원 : 최종 출력 이미지의 채널 수로, RGB 이미지를 출력하기 위해 3으로 설정합니다.
        self.cfg.supresmodule.network_capacity = 64             # 네트워크의 마지막 컨볼루션 계층 차원 : 네트워크의 복잡도를 제어하며, 값이 클수록 더 복잡한 특징을 학습할 수 있습니다.

        self.cfg.gaussianheadmodule = CN()
        self.cfg.gaussianheadmodule.num_add_mouth_points = 0    # 입 랜드마크 주변에 추가되는 점의 수 : 초기화 단계에서 입 주변의 추가 랜드마크를 생성하여 세부적인 움직임을 캡처합니다.
        self.cfg.gaussianheadmodule.exp_color_mlp = []          # 표정 색상 MLP의 차원 : 표정에 따른 색상 변화를 모델링하기 위한 다층 퍼셉트론(MLP) 구조의 차원 리스트입니다.
        self.cfg.gaussianheadmodule.pose_color_mlp = []         # 자세 색상 MLP의 차원 : 머리 자세 변화에 따른 색상 변화를 모델링하기 위한 MLP 구조의 차원 리스트입니다.
        self.cfg.gaussianheadmodule.exp_attributes_mlp = []     # 표정 속성 MLP의 차원 : 표정 속성(attribute)을 모델링하기 위한 MLP 구조의 차원 리스트입니다.
        self.cfg.gaussianheadmodule.pose_attributes_mlp = []    # 자세 속성 MLP의 차원 : 자세 속성을 모델링하기 위한 MLP 구조의 차원 리스트입니다.
        self.cfg.gaussianheadmodule.exp_deform_mlp = []         # 표정 변형 MLP의 차원 : 표정 변화에 따른 메쉬 변형을 모델링하는 MLP 구조의 차원 리스트입니다.
        self.cfg.gaussianheadmodule.pose_deform_mlp = []        # 자세 변형 MLP의 차원 : 머리 자세에 따른 메쉬 변형을 모델링하는 MLP 구조의 차원 리스트입니다.
        self.cfg.gaussianheadmodule.exp_coeffs_dim = 64         # 표정 계수의 차원 : 표정 정보를 표현하는 벡터의 차원을 지정합니다. 값이 크면 더 복잡한 표정을 표현할 수 있습니다.
        self.cfg.gaussianheadmodule.pos_freq = 4                # 위치 인코딩 빈도 : Gaussian Head에서 위치 정보를 인코딩할 때 사용하는 주파수 범위입니다.
        self.cfg.gaussianheadmodule.dist_threshold_near = 0.1   # 임계값 t1 : Gaussian Head에서 3D 가우시안과 랜드마크 간의 최소 거리로, 너무 가까운 경우를 처리합니다.
        self.cfg.gaussianheadmodule.dist_threshold_far = 0.2    # 임계값 t2 : Gaussian Head에서 3D 가우시안과 랜드마크 간의 최대 거리로, 너무 먼 경우를 처리합니다.
        self.cfg.gaussianheadmodule.deform_scale = 0.3          # 변형 스케일 팩터 : Gaussian Head의 메쉬 변형 정도를 제어합니다.
        self.cfg.gaussianheadmodule.attributes_scale = 0.05     # 속성 오프셋의 스케일 팩터 : 속성 값(예: 표정 변화) 크기를 조정하는 데 사용됩니다.

        self.cfg.recorder = CN()
        self.cfg.recorder.name = ''                             # 아바타 이름 : 재현 작업의 결과물을 구별하기 위한 이름을 지정합니다.
        self.cfg.recorder.result_path = ''                      # 시각화 결과 저장 경로 : 렌더링된 결과 이미지와 영상을 저장할 경로를 지정합니다.
