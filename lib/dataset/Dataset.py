import torch
from torch import nn
import numpy as np
import kaolin
import tqdm
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.transforms import so3_exponential_map

from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder

from lib.utils.dmtet_utils import marching_tetrahedra

class MeshDataset(Dataset):
    """
    MeshDataset:
    - 3D 메쉬 모델을 위한 데이터셋을 정의합니다.
    - 이 클래스는 3D 얼굴 모델의 이미지, 카메라 파라미터, 3D 랜드마크, 정점 데이터를 로드합니다.
    - 이 데이터셋은 훈련 중에 모델이 학습할 수 있도록 다양한 뷰에서 이미지를 제공합니다.
    """

    def __init__(self, cfg):
        """
        초기화 함수:
        - cfg: 설정(configuration) 파일을 받아 다양한 파라미터를 설정합니다. 예를 들어, 이미지 해상도, 데이터 경로 등이 포함됩니다.
        """
        super(MeshDataset, self).__init__()

        # 데이터셋에 사용할 루트 디렉터리 설정
        self.dataroot = cfg.dataroot
        self.camera_ids = cfg.camera_ids  # 사용할 카메라 ID들
        self.original_resolution = cfg.original_resolution  # 원본 이미지 해상도
        self.resolution = cfg.resolution  # 학습에 사용할 이미지 해상도
        self.num_sample_view = cfg.num_sample_view  # 샘플로 선택할 카메라 뷰의 수

        self.samples = []  # 데이터 샘플들을 저장할 리스트

        # 이미지, 파라미터, 카메라 데이터를 저장하는 폴더 경로 설정
        image_folder = os.path.join(self.dataroot, 'images')
        param_folder = os.path.join(self.dataroot, 'params')
        camera_folder = os.path.join(self.dataroot, 'cameras')
        frames = os.listdir(image_folder)  # 이미지 폴더에서 각 프레임의 폴더 이름을 가져옴
        
        self.num_exp_id = 0  # 경험(표정) ID 초기화
        for frame in frames:  # 각 프레임에 대해 데이터를 로드
            # 각 카메라에 해당하는 이미지, 마스크, 가시성, 카메라 파라미터 경로 설정
            image_paths = [os.path.join(image_folder, frame, 'image_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            mask_paths = [os.path.join(image_folder, frame, 'mask_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            visible_paths = [os.path.join(image_folder, frame, 'visible_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            camera_paths = [os.path.join(camera_folder, frame, 'camera_%s.npz' % camera_id) for camera_id in self.camera_ids]
            param_path = os.path.join(param_folder, frame, 'params.npz')
            landmarks_3d_path = os.path.join(param_folder, frame, 'lmk_3d.npy')
            vertices_path = os.path.join(param_folder, frame, 'vertices.npy')

            # 각 샘플을 리스트에 저장
            sample = (image_paths, mask_paths, visible_paths, camera_paths, param_path, landmarks_3d_path, vertices_path, self.num_exp_id)
            self.samples.append(sample)
            self.num_exp_id += 1  # 경험 ID 증가
                                  
        # 첫 번째 프레임에서 3D 랜드마크와 정점 데이터를 로드하여 초기화
        init_landmarks_3d = torch.from_numpy(np.load(os.path.join(param_folder, frames[0], 'lmk_3d.npy'))).float()
        init_vertices = torch.from_numpy(np.load(os.path.join(param_folder, frames[0], 'vertices.npy'))).float()
        init_landmarks_3d = torch.cat([init_landmarks_3d, init_vertices[::100]], 0)

        # 첫 번째 프레임의 파라미터를 로드하여 초기화
        param = np.load(os.path.join(param_folder, frames[0], 'params.npz'))
        pose = torch.from_numpy(param['pose'][0]).float()
        R = so3_exponential_map(pose[None, :3])[0]
        T = pose[None, 3:]
        S = torch.from_numpy(param['scale']).float()
        self.init_landmarks_3d_neutral = (torch.matmul(init_landmarks_3d - T, R)) / S

    def get_item(self, index):
        """
        get_item 함수:
        - 주어진 인덱스에 대한 데이터를 반환합니다.
        """
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        """
        __getitem__ 함수:
        - 데이터 샘플을 로드하여 반환합니다.
        - 각 카메라에서 이미지를 가져오고, 마스크, 가시성 데이터를 처리합니다.
        """
        sample = self.samples[index]
        
        images = []
        masks = []
        visibles = []
        views = random.sample(range(len(self.camera_ids)), self.num_sample_view)  # 무작위로 카메라 뷰 선택
        for view in views:
            image_path = sample[0][view]
            # 이미지 로드 후 리사이즈
            image = cv2.resize(io.imread(image_path), (self.resolution, self.resolution))
            image = torch.from_numpy(image / 255).permute(2, 0, 1).float()
            images.append(image)

            # 마스크 로드 후 리사이즈
            mask_path = sample[1][view]
            mask = cv2.resize(io.imread(mask_path), (self.resolution, self.resolution))[:, :, 0:1]
            mask = torch.from_numpy(mask / 255).permute(2, 0, 1).float()
            masks.append(mask)

            # 가시성 맵 로드 후 리사이즈 (가시성 맵이 없으면 기본값 1로 설정)
            visible_path = sample[2][view]
            if os.path.exists(visible_path):
                visible = cv2.resize(io.imread(visible_path), (self.resolution, self.resolution))[:, :, 0:1]
                visible = torch.from_numpy(visible / 255).permute(2, 0, 1).float()
            else:
                visible = torch.ones_like(image)
            visibles.append(visible)

        # 이미지를 텐서로 스택
        images = torch.stack(images)
        masks = torch.stack(masks)
        images = images * masks  # 마스크를 적용하여 이미지에서 보이지 않는 부분을 제거
        visibles = torch.stack(visibles)

        # 카메라 파라미터 로드
        cameras = [np.load(sample[3][view]) for view in views]
        intrinsics = torch.stack([torch.from_numpy(camera['intrinsic']).float() for camera in cameras])
        extrinsics = torch.stack([torch.from_numpy(camera['extrinsic']).float() for camera in cameras])
        
        # 카메라 내부 파라미터 조정 (원본 해상도에 맞게)
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * 2 / self.original_resolution
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * 2 / self.original_resolution - 1
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * 2 / self.original_resolution
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * 2 / self.original_resolution - 1

        # 파라미터 파일에서 자세, 스케일, 표정 계수 로드
        param_path = sample[4]
        param = np.load(param_path)
        pose = torch.from_numpy(param['pose'][0]).float()
        scale = torch.from_numpy(param['scale']).float()
        exp_coeff = torch.from_numpy(param['exp_coeff'][0]).float()

        # 3D 랜드마크 및 정점 데이터 로드
        landmarks_3d_path = sample[5]
        landmarks_3d = torch.from_numpy(np.load(landmarks_3d_path)).float()
        vertices_path = sample[6]
        vertices = torch.from_numpy(np.load(vertices_path)).float()
        landmarks_3d = torch.cat([landmarks_3d, vertices[::100]], 0)

        exp_id = sample[7]

        # 데이터 반환
        return {
                'images': images,
                'masks': masks,
                'visibles': visibles,
                'pose': pose,
                'scale': scale,
                'exp_coeff': exp_coeff,
                'landmarks_3d': landmarks_3d,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'exp_id': exp_id}

    def __len__(self):
        """
        __len__ 함수:
        - 데이터셋의 크기를 반환합니다.
        """
        return len(self.samples)

