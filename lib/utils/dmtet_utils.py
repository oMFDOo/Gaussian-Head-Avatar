import torch

# 삼각형 테이블: 각 테트라헤드럴의 점들로 구성된 삼각형 정보를 정의
triangle_table = torch.tensor([
    [-1, -1, -1, -1, -1, -1],  # 삼각형이 없는 경우
    [1, 0, 2, -1, -1, -1],     # 하나의 삼각형 정의
    [4, 0, 3, -1, -1, -1],     # 다른 하나의 삼각형 정의
    [1, 4, 2, 1, 3, 4],        # 두 개의 삼각형 정의
    [3, 1, 5, -1, -1, -1],
    [2, 3, 0, 2, 5, 3],
    [1, 4, 0, 1, 5, 4],
    [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1],
    [4, 1, 0, 4, 5, 1],
    [3, 2, 0, 3, 5, 2],
    [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1],
    [3, 0, 4, -1, -1, -1],
    [2, 0, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
], dtype=torch.long)

# 테트라헤드럴 내에서 삼각형의 수를 미리 정의한 테이블
num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)

# 테트라헤드럴의 기본 엣지 정의 (점들의 쌍)
base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)

# 4개의 점을 2의 거듭제곱으로 매핑 (비트마스크 활용)
v_id = torch.pow(2, torch.arange(4, dtype=torch.long))

def _sort_edges(edges):
    """
    각 엣지(2개의 점으로 구성)의 순서를 정렬하여 표준화.
    입력:
        edges: (E, 2) 크기의 텐서, 각 행이 하나의 엣지
    출력:
        정렬된 엣지들
    """
    with torch.no_grad():
        # 첫 번째 점이 두 번째 점보다 큰지 확인하여 순서를 결정
        order = (edges[:, 0] > edges[:, 1]).long().unsqueeze(dim=1)
        a = torch.gather(input=edges, index=order, dim=1)
        b = torch.gather(input=edges, index=1 - order, dim=1)
    return torch.stack([a, b], -1)

def _unbatched_marching_tetrahedra(vertices, features, tets, sdf, return_tet_idx):
    """
    단일 배치에 대한 Marching Tetrahedra 알고리즘 구현.
    
    입력:
        vertices: 점들의 좌표
        features: 각 점의 특성
        tets: 테트라헤드럴의 구성 (점들의 인덱스)
        sdf: 각 점의 Signed Distance Field 값
        return_tet_idx: 삼각형이 추출된 테트라헤드럴의 인덱스를 반환할지 여부
    출력:
        생성된 점들, 점의 특성, 삼각형의 구성, (선택적으로) 테트라헤드럴의 인덱스
    """
    device = vertices.device

    with torch.no_grad():
        # 점이 내부(occ_n=True)인지 외부(occ_n=False)인지 결정
        occ_n = sdf > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)  # 테트라헤드럴의 점들의 내부 여부
        occ_sum = torch.sum(occ_fx4, -1)  # 각 테트라헤드럴에서 내부 점들의 개수 계산

        # 유효한 테트라헤드럴: 내부 점이 0보다 많고 4보다 적은 경우
        valid_tets = (occ_sum > 0) & (occ_sum < 4)
        occ_sum = occ_sum[valid_tets]

        # 유효한 엣지 찾기
        all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)
        all_edges = _sort_edges(all_edges)
        unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

        # 엣지 마스크 생성
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
        idx_map = mapping[idx_map]

        # 보간법(interpolation)을 통한 새로운 점 계산
        interp_v = unique_edges[mask_edges]
    edges_to_interp = vertices[interp_v.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)
    edges_to_interp_sdf[:, -1] *= -1

    edges_to_interp_features = features[interp_v.reshape(-1)].reshape(-1, 2, features.shape[1])

    # 보간법 계산
    denominator = edges_to_interp_sdf.sum(1, keepdim=True)
    edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
    verts = (edges_to_interp * edges_to_interp_sdf).sum(1)
    vert_features = (edges_to_interp_features * edges_to_interp_sdf).sum(1)

    idx_map = idx_map.reshape(-1, 6)

    # 테트라헤드럴의 상태 인덱스 계산
    tetindex = (occ_fx4[valid_tets] * v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table.to(device)[tetindex]
    triangle_table_device = triangle_table.to(device)

    # 삼각형의 구성 생성
    faces = torch.cat((
        torch.gather(input=idx_map[num_triangles == 1], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
        torch.gather(input=idx_map[num_triangles == 2], dim=1,
                     index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
    ), dim=0)

    if return_tet_idx:
        tet_idx = torch.arange(tets.shape[0], device=device)[valid_tets]
        tet_idx = torch.cat((tet_idx[num_triangles == 1], tet_idx[num_triangles ==
                            2].unsqueeze(-1).expand(-1, 2).reshape(-1)), dim=0)
        return verts, vert_features, faces, tet_idx
    return verts, vert_features, faces


def marching_tetrahedra(vertices, features, tets, sdf, return_tet_idx=False):
    r"""테트라헤드론 그리드에 인코딩된 이산 Signed Distance Field (SDF)를 Marching Tetrahedra 알고리즘을 사용하여 삼각형 메시로 변환합니다. 
    출력된 표면은 입력된 정점 위치와 SDF 값에 대해 미분 가능하며, 
    더 자세한 내용과 예시 사용법은 'Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis' 논문(NeurIPS 2021)에서 확인할 수 있습니다.

    Args:
        vertices (torch.tensor): 테트라헤드론 메시의 배치된 정점들로, 형태는 :math:`(\text{batch_size}, \text{num_vertices}, 3)`입니다.
        tets (torch.tensor): 배치되지 않은 테트라헤드론 메시의 토폴로지로, 형태는 :math:`(\text{num_tetrahedrons}, 4)`입니다.
        sdf (torch.tensor): 각 정점의 SDF 값을 지정하는 배치된 SDF로, 형태는 :math:`(\text{batch_size}, \text{num_vertices})`입니다.
        return_tet_idx (optional, bool): True일 경우, 각 면이 추출된 테트라헤드론의 인덱스를 반환합니다. 기본값은 False입니다.

    Returns:
        (list[torch.Tensor], list[torch.LongTensor], (optional) list[torch.LongTensor]): 

            - 각 테트라헤드론 그리드에서 변환된 메시의 정점 목록.
            - 각 테트라헤드론 그리드에서 변환된 메시의 면 목록.
            - 각 면이 추출된 테트라헤드론의 인덱스 목록(옵션).

    예시:
        >>> vertices = torch.tensor([[[0, 0, 0],
        ...               [1, 0, 0],
        ...               [0, 1, 0],
        ...               [0, 0, 1]]], dtype=torch.float)
        >>> tets = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        >>> sdf = torch.tensor([[-1., -1., 0.5, 0.5]], dtype=torch.float)
        >>> verts_list, faces_list, tet_idx_list = marching_tetrahedra(vertices, tets, sdf, True)
        >>> verts_list[0]
        tensor([[0.0000, 0.6667, 0.0000],
                [0.0000, 0.0000, 0.6667],
                [0.3333, 0.6667, 0.0000],
                [0.3333, 0.0000, 0.6667]])
        >>> faces_list[0]
        tensor([[3, 0, 1],
                [3, 2, 0]])
        >>> tet_idx_list[0]
        tensor([0, 0])

    .. _테트라헤드론 셀을 사용한 등가값 표면의 삼각화 효율적인 방법:
        https://search.ieice.org/bin/summary.php?id=e74-d_1_214

    .. _Deep Marching Tetrahedra: 고해상도 3D 형태 합성을 위한 하이브리드 표현:
        https://arxiv.org/abs/2111.04276
    """
    """
    Marching Tetrahedra 알고리즘 구현.

    입력:
        vertices: 점들의 좌표
        features: 각 점의 특성
        tets: 테트라헤드럴의 구성 (점들의 인덱스)
        sdf: Signed Distance Field 값
        return_tet_idx: 삼각형이 추출된 테트라헤드럴의 인덱스를 반환할지 여부
    출력:
        각 배치에 대해 생성된 점들, 삼각형, 선택적으로 테트라헤드럴 인덱스
    """
    list_of_outputs = [
        _unbatched_marching_tetrahedra(vertices[b], features[b], tets, sdf[b], return_tet_idx)
        for b in range(vertices.shape[0])
    ]
    return list(zip(*list_of_outputs))
