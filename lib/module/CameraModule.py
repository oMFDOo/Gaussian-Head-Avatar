import torch
import kaolin
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class CameraModule():
    """
    3D 카메라 모듈:
    - 3D 데이터를 카메라 시점으로 변환, 투영, 렌더링하는 기능을 제공합니다.
    - DIB-R 방식 및 Gaussian Rasterization 방식을 지원합니다.
    """
    def __init__(self):
        """
        CameraModule 초기화:
        - 배경색과 스케일 수정 값을 기본으로 설정합니다.
        """
        self.bg_color = torch.tensor([1.0] * 32).float()  # 배경 색상 설정 (기본 흰색, 1.0)
        self.scale_modifier = 1.0  # 스케일 조정 값 (디폴트 1.0)

    def perspective_camera(self, points, camera_proj):
        """
        3D 공간의 점들을 원근 투영(Perspective Projection)을 통해 2D로 변환합니다.
        - points: 3D 점 좌표 텐서 (BxNx3) -> 배치 크기(B)와 점 개수(N)
        - camera_proj: 카메라 투영 행렬 텐서 (Bx4x4)
        반환값:
            - 2D로 투영된 점 좌표 (BxNx2)
        """
        # 3D 점들을 카메라 투영 행렬로 변환
        projected_points = torch.bmm(points, camera_proj.permute(0, 2, 1))  # 행렬 곱
        # Homogeneous 좌표계를 Cartesian 좌표계로 변환
        projected_2d_points = projected_points[:, :, :2] / projected_points[:, :, 2:3]  # Z축으로 나누어 2D로 투영
        return projected_2d_points

    def prepare_vertices(self, vertices, faces, camera_proj, camera_rot=None, camera_trans=None,
                         camera_transform=None):
        """
        3D 정점(vertices)을 카메라 좌표계로 변환하고 투영합니다.
        - vertices: 3D 정점 좌표 텐서 (Nx3)
        - faces: 면(Face)을 정의하는 텐서 (Fx3)
        - camera_proj: 카메라 투영 행렬
        - camera_rot: 카메라의 회전 행렬
        - camera_trans: 카메라의 평행 이동 벡터
        - camera_transform: 카메라 변환 행렬 (회전 및 평행 이동 포함, 둘 중 하나만 필요)
        반환값:
            - face_vertices_camera: 카메라 좌표계의 면 정점 좌표
            - face_vertices_image: 투영된 2D 면 정점 좌표
            - face_normals: 면의 법선 벡터
        """
        if camera_transform is None:
            # 변환 행렬이 없으면 회전 및 평행 이동을 개별적으로 적용
            assert camera_trans is not None and camera_rot is not None, \
                "camera_transform 또는 camera_trans, camera_rot 중 하나는 필수"
            vertices_camera = kaolin.render.camera.rotate_translate_points(vertices, camera_rot, camera_trans)
        else:
            # 변환 행렬이 주어지면 이를 사용하여 정점을 카메라 좌표계로 변환
            assert camera_trans is None and camera_rot is None, \
                "camera_transform 사용 시 camera_trans와 camera_rot는 None이어야 합니다."
            # Homogeneous 좌표로 변환 (1을 추가)
            padded_vertices = torch.nn.functional.pad(vertices, (0, 1), mode='constant', value=1.)
            vertices_camera = (padded_vertices @ camera_transform)

        # 카메라 좌표계를 2D로 투영
        vertices_image = self.perspective_camera(vertices_camera, camera_proj)

        # 면(Face) 별 정점 좌표 계산
        face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, faces)  # 카메라 좌표
        face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_image, faces)  # 투영된 2D 좌표
        # 면의 노멀(Normal) 벡터 계산 (단위 벡터)
        face_normals = kaolin.ops.mesh.face_normals(face_vertices_camera, unit=True)

        return face_vertices_camera, face_vertices_image, face_normals

    def render(self, data, resolution):
        """
        DIB-R 방식으로 3D 모델을 렌더링합니다.
        - data: 렌더링할 데이터 딕셔너리 (정점, 면, 색상 등 포함)
        - resolution: 출력 이미지 해상도
        반환값: 렌더링 결과를 포함하는 데이터 딕셔너리
        """
        verts_list = data['verts_list']  # 정점 리스트
        faces_list = data['faces_list']  # 면 리스트
        verts_color_list = data['verts_color_list']  # 정점의 색상 리스트

        B = len(verts_list)  # 배치 크기

        # 렌더링 결과를 저장할 리스트 초기화
        render_images = []  # 렌더링된 이미지
        render_soft_masks = []  # 부드러운 마스크
        render_depths = []  # 깊이 맵
        render_normals = []  # 노멀 맵
        face_normals_list = []  # 면의 노멀 벡터

        for b in range(B):
            # 카메라 관련 정보 로드
            intrinsics = data['intrinsics'][b]  # 카메라 내부 파라미터 (투영 행렬)
            extrinsics = data['extrinsics'][b]  # 카메라 외부 파라미터 (변환 행렬)
            camera_proj = intrinsics  # 투영 행렬
            camera_transform = extrinsics.permute(0, 2, 1)  # 변환 행렬 (회전 및 평행 이동 포함)

            # 정점, 면, 색상을 배치 크기만큼 복제
            verts = verts_list[b].unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)  # 정점
            faces = faces_list[b]  # 면
            verts_color = verts_color_list[b].unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)  # 색상
            faces_color = verts_color[:, faces]  # 면의 색상

            # 정점 및 면 준비
            face_vertices_camera, face_vertices_image, face_normals = self.prepare_vertices(
                verts, faces, camera_proj, camera_transform=camera_transform
            )

            # Y축 반전 처리 (화면 좌표계와 맞추기 위해)
            face_vertices_image[:, :, :, 1] = -face_vertices_image[:, :, :, 1]
            face_normals[:, :, 1:] = -face_normals[:, :, 1:]

            # 면 속성 정의
            face_attributes = [
                faces_color,  # 면 색상
                torch.ones((faces_color.shape[0], faces_color.shape[1], 3, 1), device=verts.device),  # 하드 마스크
                face_vertices_camera[:, :, :, 2:],  # 깊이 정보
                face_normals.unsqueeze(-2).repeat(1, 1, 3, 1),  # 면의 노멀
            ]

            # DIB-R 방식으로 렌더링 수행
            image_features, soft_masks, face_idx = kaolin.render.mesh.dibr_rasterization(
                resolution, resolution, -face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1],
                rast_backend='cuda')

            # 렌더링된 이미지 및 속성 저장
            images, masks, depths, normals = image_features
            images = torch.clamp(images * masks, 0., 1.)  # 유효 영역 클램핑
            depths = (depths * masks)  # 깊이 맵 마스킹
            normals = (normals * masks)  # 노멀 맵 마스킹

            # 결과 저장
            render_images.append(images)
            render_soft_masks.append(soft_masks)
            render_depths.append(depths)
            render_normals.append(normals)
            face_normals_list.append(face_normals)

        # 결과를 데이터 딕셔너리에 추가
        render_images = torch.stack(render_images, 0)
        render_soft_masks = torch.stack(render_soft_masks, 0)
        render_depths = torch.stack(render_depths, 0)
        render_normals = torch.stack(render_normals, 0)

        data['render_images'] = render_images
        data['render_soft_masks'] = render_soft_masks
        data['render_depths'] = render_depths
        data['render_normals'] = render_normals
        data['verts_list'] = verts_list
        data['faces_list'] = faces_list
        data['face_normals_list'] = face_normals_list

        return data

    def render_gaussian(self, data, resolution):
        """
        Gaussian Rasterization 방식으로 3D 모델을 렌더링합니다.
        - data: 렌더링할 데이터 딕셔너리
        - resolution: 출력 이미지 해상도
        반환값: 렌더링 결과를 포함하는 데이터 딕셔너리
        """
        B = data['xyz'].shape[0]  # 배치 크기
        xyz = data['xyz']  # 3D 정점 좌표
        colors_precomp = data['color']  # 정점 색상
        opacity = data['opacity']  # 불투명도
        scales = data['scales']  # 정점 크기
        rotations = data['rotation']  # 회전 정보
        fovx = data['fovx']  # X축 시야각
        fovy = data['fovy']  # Y축 시야각

        # 기타 카메라 변환 정보
        world_view_transform = data['world_view_transform']
        full_proj_transform = data['full_proj_transform']
        camera_center = data['camera_center']

        # 화면 좌표 초기화 (그라디언트를 계산하기 위해 필요)
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device)

        render_images = []
        radii = []

        for b in range(B):
            # 시야각 계산
            tanfovx = math.tan(fovx[b] * 0.5)
            tanfovy = math.tan(fovy[b] * 0.5)

            # Gaussian Rasterization 설정
            raster_settings = GaussianRasterizationSettings(
                image_height=int(resolution),
                image_width=int(resolution),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=self.bg_color.to(xyz.device),
                scale_modifier=self.scale_modifier,
                viewmatrix=world_view_transform[b],
                projmatrix=full_proj_transform[b],
                sh_degree=0,
                campos=camera_center[b],
                prefiltered=False,
                debug=False
            )

            # Rasterizer 객체 생성
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            # 3D 및 화면 좌표를 사용해 Gaussian Rasterization 수행
            render_images_b, radii_b = rasterizer(
                means3D=xyz[b],
                means2D=screenspace_points[b],
                colors_precomp=colors_precomp[b],
                opacities=opacity[b],
                scales=scales[b],
                rotations=rotations[b])

            render_images.append(render_images_b)
            radii.append(radii_b)

        # 결과를 데이터 딕셔너리에 추가
        render_images = torch.stack(render_images)
        radii = torch.stack(radii)
        data['render_images'] = render_images
        data['viewspace_points'] = screenspace_points
        data['visibility_filter'] = radii > 0  # 화면에 보이는 가우시안 필터
        data['radii'] = radii

        return data
