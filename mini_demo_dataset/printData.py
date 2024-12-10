import numpy as np

print()
print('==========================')
print('> camera_220700191.npz')
print('==========================')
# 카메라 .npz 파일 로드
camera_file = 'camera_220700191.npz'
camera_path = './031/cameras/0000/' + camera_file
camera_data = np.load(camera_path)
print()
# 파일 내 변수 목록 출력
print(camera_data.files)

# 특정 변수 출력
# extrinsic (외적) 파라미터 : 4x4 변환 행렬로 구성되어 있으며, 카메라의 월드 좌표계를 카메라 좌표계로 변환하는 데 사용
print(camera_data['extrinsic'])

# intrinsic (내적) 파라미터: 카메라의 초점 거리(focal length)와 이미지 센터의 위치를 나타내며, 카메라의 렌즈 특성에 따라 결정
# (fx, 0, cx): fx는 초점 거리(focal length)이며, cx는 이미지 센터의 x 좌표
# (0, fy, cy): fy는 초점 거리의 세로 방향 값이며, cy는 이미지 센터의 y 좌표
# (0, 0, 1): 이는 동차 좌표를 위한 값
print(camera_data['intrinsic'])

print()
print('==========================')
print('> lmk_220700191.npy')
print('==========================')
# 랜드마크 .npy 파일 로드 
lmk_file = 'lmk_220700191.npy'
lmk_path = './031/landmarks/0000/' + lmk_file
lmk_data = np.load(lmk_path)
print()
# 전체 출력 : 68개의 랜드마크 포인트
print(lmk_data)
print(len(lmk_data))

print()
print('==========================')
print('> params.npz')
print('==========================')
params_file = 'params.npz'
params_path = './031/params/0000/' + params_file
params_data = np.load(params_path)
print()
# 파일 내 변수 목록 출력
print(params_data.files)

# 얼굴 모델의 정체성(Identity)을 나타내는 계수들입니다. 
# 이 값들은 얼굴의 기본적인 형상 (예: 얼굴 크기, 턱선, 이마의 모양 등)을 정의합니다. 
# 각 id_coeff는 얼굴의 특정 부분의 형태를 어떻게 변형시킬지를 결정하는 벡터로, 얼굴 모델의 "기본 얼굴"을 변형하는 데 사용됩니다.
print('id_coeff')
print(params_data['id_coeff'])

# exp_coeff는 얼굴의 표정(Expression)을 나타내는 계수들입니다.
# 이 계수들은 특정 얼굴 표정 (예: 웃음, 찡그린 얼굴 등)을 만들기 위한 변형을 정의합니다.
# exp_coeff 값은 얼굴 근육의 움직임을 모델링하여 표정을 재현하는 데 사용됩니다
print('exp_coeff')
print(params_data['exp_coeff'])

# 3D 모델의 크기를 조정하는 계수입니다.
# 모델의 크기 조정에는 비율이 사용되며, 이 값은 모델을 크거나 작게 만드는 데 사용됩니다.
print('scale')
print(params_data['scale'])

# 모델의 위치와 회전을 나타내는 매개변수입니다.
# 즉, 얼굴이나 3D 모델이 3D 공간에서 어떻게 배치되는지, 특정 각도에서 보이는지 등을 정의합니다.
# pose는 일반적으로 모델의 회전 행렬을 나타냅니다.
print('pose')
print(params_data['pose'])

print()
print('==========================')
print('> lmk_3d.npy')
print('==========================')
params_file = 'lmk_3d.npy'
params_path = './031/params/0000/' + params_file
params_data = np.load(params_path)
print()
# 전체 출력 : 66개 랜드마크
print(params_data)
print(len(params_data))

print()
print('==========================')
print('> vertices.npy')
print('==========================')
params_file = 'vertices.npy'
params_path = './031/params/0000/' + params_file
params_data = np.load(params_path)
print()
# 전체 출력 : 
print(params_data)

# dataroot: 데이터셋의 최상위 경로입니다. mini_demo_dataset/036은 데이터셋이 위치한 디렉토리입니다. 이 경로는 이미지와 파라미터 파일들이 포함된 폴더입니다.
# image_files: 얼굴 이미지를 저장한 파일 경로입니다. 여기서 'images/*/image_222200037.jpg'는 이미지가 위치한 경로를 나타내며, *은 이미지의 하위 폴더들이 포함될 수 있다는 뜻입니다.
# param_files: 얼굴 모델의 파라미터 파일 경로입니다. 여기서 'params/*/params.npz'는 각 이미지와 관련된 3DMM 파라미터들을 포함하는 파일들의 경로입니다.
# camera_path: 카메라 파라미터 파일 경로입니다. 'mini_demo_dataset/031/cameras/0000/camera_222200037.npz'는 카메라의 내적 및 외적 파라미터를 포함한 .npz 파일 경로입니다. 이 파일은 카메라의 위치와 회전 정보를 제공합니다.
# pose_code_path: 3D 얼굴 모델의 포즈 코드가 포함된 파일 경로입니다. 'mini_demo_dataset/031/params/0000/params.npz'는 얼굴의 포즈 정보를 담고 있는 .npz 파일입니다.
# freeview: True로 설정된 경우, 자유로운 카메라 뷰에서 얼굴을 생성할 수 있도록 합니다. 이 값이 True이면, update_camera를 통해 매 프레임마다 카메라 위치를 업데이트할 수 있습니다.
# resolution: 입력 이미지의 해상도입니다. 2048로 설정되어 있으므로, 모델은 2048x2048 크기의 이미지를 사용하여 얼굴을 생성할 것입니다.
# original_resolution: 원본 이미지 해상도를 설정합니다. 이 값이 2048로 설정되어 있으므로, 원본 이미지의 해상도가 2048x2048임을 의미합니다.