import copy

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm
from .dataset import load_reference_datasets, load_query_datasets
from .models import CLIPModel
from .utils import (
    blend_images,
    find_matches,
    generate_annotation_distribution,
    generate_super_points,
    get_image_embeddings,
    get_reference_annotations,
    get_spot_embeddings,
    load_model,
    normalize_celltype,
    process_image_to_adata,
    get_query_coordinates,
    calculate_coord_similarity,
    normalize_tensor_to_neg1_1,
    find_k_nearest_averages_with_threshold,
    generate_mask,
    softmax,
    fusion
)


def infer_base(
    reference_spot_ebd,
    reference_annotations,
    query_img_ebd,
    similarity_matrix=None,
    num_neighbors=5,
    topk=5,
    mode="basic",
    annotation_type="continuous",
):
    """
    根据query_img_ebd，找到reference_spot_ebd中Top-K最相似的嵌入，并根据模式进行处理。

    参数:
    - query_img_ebd (torch.Tensor): 查询图片的嵌入向量，维度为(M, D)
    - reference_spot_ebd (torch.Tensor): 关键点嵌入向量，维度为(N, D)
    - reference_annotations (torch.Tensor): 关键点的注解向量，维度为(N, F)
    - similarity_matrix (torch.Tensor): 空间相似性矩阵，维度为(M, M)，只在'advanced'模式下使用
    - num_neighbors (int): 低相似度情况下选择的最相似的img_ebd数量，只在'advanced'模式下使用
    - topk (int): 每个查询向量需要找到的最相似的Top-K个向量
    - mode (str): 'basic' 或 'advanced'，指定处理的模式

    返回:
    - inferred_spot_embeddings (torch.Tensor): 推断出的嵌入向量，维度为(M, D)
    - inferred_spot_annotations (torch.Tensor): 推断出的注解向量，维度为(M, F)
    """
    # 计算余弦相似度

    # 确保所有输入均为torch.Tensor
    if isinstance(reference_spot_ebd, np.ndarray):
        reference_spot_ebd = torch.from_numpy(reference_spot_ebd)
    if isinstance(reference_annotations, np.ndarray):
        reference_annotations = torch.from_numpy(reference_annotations)
    if isinstance(query_img_ebd, np.ndarray):
        query_img_ebd = torch.from_numpy(query_img_ebd)
    if similarity_matrix is not None and isinstance(similarity_matrix, np.ndarray):
        similarity_matrix = torch.from_numpy(similarity_matrix)
    if annotation_type == "discrete":
        topk = 1  # only match the most similar annotation

    print("Querying...")
    similarity = torch.matmul(query_img_ebd, reference_spot_ebd.t())
    similarity = normalize_tensor_to_neg1_1(similarity)
    # 每个点取tok，则共topk*len(query_img_ebd)个分数，比例为topk*len(query_img_ebd)/len(query_img_ebd)**2=topk/len(query_img_ebd)
    # 因此，低于这个分位点，则认为相似度较低

    top_k_values, top_k_indices = torch.topk(similarity, topk, dim=1)

    # 初始化结果张量
    inferred_spot_embeddings = torch.zeros_like(query_img_ebd)
    inferred_spot_annotations = torch.zeros(
        query_img_ebd.shape[0], reference_annotations.shape[1]
    )

    if mode == "basic":
        # 基础模式，直接计算Top-K平均
        for i in range(query_img_ebd.shape[0]):
            top_k_reference_spot_ebd = reference_spot_ebd[top_k_indices[i]]
            inferred_spot_embeddings[i] = torch.mean(top_k_reference_spot_ebd, dim=0)
            top_k_reference_annotations = reference_annotations[top_k_indices[i]]
            inferred_spot_annotations[i] = torch.mean(top_k_reference_annotations, dim=0)
    elif mode == "advanced" and similarity_matrix is not None:
        # 高级模式，考虑低相似度的特殊处理
        query_img_similarity = torch.matmul(query_img_ebd, query_img_ebd.t())
        mask = query_img_similarity < 0.8
        similarity_matrix[mask] = -1
        similarity_ = similarity.flatten()
        sampled_indices = torch.randint(0, similarity_.size(0), (10000,))
        similarity_ = similarity_[sampled_indices]
        threshold = similarity_.quantile(1 - topk / len(query_img_ebd))
        for i in range(query_img_ebd.shape[0]):
            valid_indices = top_k_indices[i][top_k_values[i] > threshold]
            if len(valid_indices) > 0:
                top_k_reference_spot_ebd = reference_spot_ebd[valid_indices]
                inferred_spot_embeddings[i] = torch.mean(top_k_reference_spot_ebd, dim=0)
                top_k_reference_annotations = reference_annotations[valid_indices]
                inferred_spot_annotations[i] = torch.mean(top_k_reference_annotations, dim=0)
            else:
                # 使用空间相似性矩阵找到最空间位置相似且图像相似的Q个indices
                _, indices = torch.topk(similarity_matrix[i], num_neighbors)
                query_related = query_img_ebd[indices]
                sim = torch.matmul(query_related, reference_spot_ebd.t())
                values, indices = torch.topk(sim, topk, dim=1)
                top_k_reference_spot_ebd = reference_spot_ebd[indices.flatten()]
                inferred_spot_embeddings[i] = torch.mean(top_k_reference_spot_ebd, dim=0)
                top_k_reference_annotations = reference_annotations[indices.flatten()]
                inferred_spot_annotations[i] = torch.mean(top_k_reference_annotations, dim=0)
                
    inferred_spot_embeddings = torch.nn.functional.normalize(inferred_spot_embeddings, p=2, dim=1)
    return (
        inferred_spot_embeddings.cpu().numpy(),
        inferred_spot_annotations.cpu().numpy(),
    )


def infer(
    adata,
    topk=20,
    lower_perc=0.1,
    mode="advanced",
    device="cuda:0",
    annotation_type="continuous",
):
    """
    Infer cell type, or status of spots.

    Args:
    - adata (AnnData): AnnData object containing the data.
    - method (str): Method for inference, options are "simple", "average", "weighted_average".
    - topk (int): Number of top matches to consider for inference.
    """

    query_img_ebd = get_image_embeddings(adata, "query", device)
    reference_spot_ebd = get_spot_embeddings(adata, "reference", device, annotation_type)
    adata.uns["reference_spot_embeddings"] = reference_spot_ebd
    reference_annotations = get_reference_annotations(adata, annotation_type)
    coordinates = get_query_coordinates(adata)
    coord_similarity = calculate_coord_similarity(coordinates)

    num_neighbors = query_img_ebd.shape[0] // 100 + 1
    inferred_spot_embeddings, inferred_spot_annotations = infer_base(
        reference_spot_ebd,
        reference_annotations,
        query_img_ebd,
        similarity_matrix=coord_similarity,
        num_neighbors=num_neighbors,
        topk=topk,
        mode=mode,
        annotation_type=annotation_type,
    )

    adata.obsm["inferred_spot_embeddings"] = inferred_spot_embeddings
    # inferred_spot_annotations = softmax(inferred_spot_annotations)
    inferred_spot_annotations = normalize_celltype(
        inferred_spot_annotations, lower_perc
    )  # 0 for no truncation, 1 for all truncation
    
    adata.obsm["inferred_spot_annotations"] = inferred_spot_annotations


def super_infer(
    adata, scale=2, distance_threshold=160, topk=20, lower_perc=0.1, mode="basic",annotation_type="continuous", device="cuda:0",
):
    """
    Infer cell type, or status of spots with super resolution.

    Args:
    - adata (AnnData): AnnData object containing the data.
    - scale (int): Scale factor for super resolution.
    - distance_threshold (int): Distance threshold for super points generation.
    - method (str): Method for inference, options are "simple", "average", "weighted_average".
    - topk (int): Number of top matches to consider for inference.
    """

    adata_super = adata.copy()
    spot_orginal_num = adata.shape[0]
    size = int(np.sqrt(spot_orginal_num))

    points = adata.obsm["spatial"].copy()
    # 创建一个更密集的网格，这次仅关注坐标生成
    super_points = generate_super_points(points, size, scale, distance_threshold)

    # 重新初始化.X矩阵以匹配新的观测数量
    new_X_shape = (super_points.shape[0], adata.X.shape[1])  # 假设保留相同的特征数量
    X = np.random.randn(*new_X_shape)  # 以随机数重新初始化，或其他初始化方法
    names = [f"{int(x)}x{int(y)}" for x, y in super_points]

    df = pd.DataFrame(X, columns=adata.var_names.tolist(), index=names)
    adata_super = sc.AnnData(df)
    # 更新.obsm以存储新的坐标
    adata_super.obsm["spatial"] = super_points  # 可以选择合适的键名
    adata_super.uns["spatial"] = copy.deepcopy(adata.uns["spatial"])
    adata_super.uns["spatial"]["test"]["scalefactors"]["spot_diameter_fullres"] = (
        adata.uns["spatial"]["test"]["scalefactors"]["spot_diameter_fullres"]
        / (2 + scale)
        * 2
    )

    adata_super.model = copy.deepcopy(adata.model)
    load_query_datasets(adata_super)
    image_embeddings = get_image_embeddings(adata_super, name="query", device=device)

    adata_super.uns["reference_spot_embeddings"] = adata.uns["reference_spot_embeddings"]
    adata_super.uns["reference_annotations"] = adata.uns["reference_annotations"]
    reference_spot_ebd = adata_super.uns["reference_spot_embeddings"]
    reference_annotations = adata_super.uns["reference_annotations"]
    query_img_ebd = image_embeddings

    coordinates = get_query_coordinates(adata_super)
    coord_similarity = calculate_coord_similarity(coordinates)
    num_neighbors = query_img_ebd.shape[0] // 100 + 1
    inferred_spot_embeddings, inferred_spot_annotations = infer_base(
        reference_spot_ebd,
        reference_annotations,
        query_img_ebd,
        similarity_matrix=coord_similarity,
        num_neighbors=num_neighbors,
        topk=topk,
        mode=mode,
        annotation_type=annotation_type
    )

    adata_super.obsm["inferred_spot_embeddings"] = inferred_spot_embeddings
    # inferred_spot_annotations = softmax(inferred_spot_annotations)
    inferred_spot_annotations = normalize_celltype(
        inferred_spot_annotations, lower_perc
    )  # 0 for no truncation, 1 for all truncation

    adata_super.obsm["inferred_spot_annotations"] = inferred_spot_annotations
    adata_super.uns["annotation_list"] = adata.uns["annotation_list"]
    return adata_super


def infer_from_image(
    image_path,
    model_path,
    reference_adata_paths=None,
    model=CLIPModel(),
    grid_size=112,
    density_threshold=0.9,
    mode="basic",
    topk=20,
    lower_perc=0.2,
    annotation_type="discrete",
    annotation_list = None,
    device='cuda:0'
):
    """
    Infer cell type or status of spots from an image.

    Args:
    - image_path (str): Path to the input image.
    - reference_adata_paths (list): List of paths to reference datasets.
    - model_path (str): Path to the pre-trained model.
    - method (str): Method for inference, options are "simple", "average", "weighted_average".
    - topk (int): Number of top matches to consider for inference.

    Returns:
    - adata (AnnData): Anndata object containing inferred spot annotations.
    """
    assert (annotation_list is not None) or (len(reference_adata_paths)>0), "Please input annotation information!"
    # Step 1: Load the input image
    # Assuming you have a function to process the image and create spots
    adata = process_image_to_adata(
        image_path, grid_size, density_threshold
    )  # You need to implement this function

    # Step 2: Load reference datasets and concatenate them into a single AnnData object
    if reference_adata_paths is None:
        adata.uns['annotation_list'] = annotation_list
    else:
        load_reference_datasets(adata, dataset_paths=reference_adata_paths)

    load_model(adata, model_path=model_path, model=model)
    load_query_datasets(adata)

    infer(adata=adata, topk=topk, lower_perc=lower_perc, mode=mode, annotation_type=annotation_type, device=device)

    return adata


def image_generation(adata, pesudo_annotation, topk=10, mode = 'GFF', device="cuda:0"):
    """
    Generate image according to simulation spot status.
    """
    model = adata.model
    model = model.to(device)
    model.eval()
    pesudo_annotation = torch.tensor(pesudo_annotation).float()
    pesudo_annotation = pesudo_annotation.to(next(model.parameters()).device)
    spot_embeddings_query = (
        model.spot_projection(pesudo_annotation).cpu().detach().numpy()
    )
    assert "reference_dataloaders" in adata.uns.keys(), "Run load_reference_datasets first!"
    if "reference_img_embeddings" not in adata.uns.keys():
        img_embeddings_reference = get_image_embeddings(adata, "reference", device)
    else:
        img_embeddings_reference = adata.uns["reference_img_embeddings"]
    topsimilarity, indices = find_matches(
        img_embeddings_reference, spot_embeddings_query, topk=topk
    )
    dataset = adata.uns["reference_dataset"]
    img_list = []
    print("Fusioning patches...")
    for idx in tqdm(indices):
        batch_images = torch.stack([dataset[i]["image"].permute(1, 2, 0) for i in idx])
        if mode=='GFF':
            for i, img_ in enumerate(batch_images):
                if i==0:
                    img_fusion = batch_images[i].cpu().detach().numpy().astype(int)
                else:
                    img_fusion = fusion(img_fusion, batch_images[i].cpu().detach().numpy().astype(int))
                
            generated_img = torch.as_tensor(img_fusion)
        elif mode=='mean':
            
            generated_img = batch_images.mean(0)
        img_list.append(generated_img.int().squeeze())
    pred_images = torch.stack(img_list, axis=0)

    return pred_images


def generation_pesudo_tissue(
    adata,
    image_size=(10000, 10000),
    step=56,
    layer_size=1000,
    num_layers=4,
    patch_size=224,
    ord=2,
):
    # 构建网格坐标；
    # 根据形状构建渐变的annotation矩阵；
    # 根据annotation，进行每块的生成
    # 将所有块进行合成；(通过旋转让图片更连续？)

    # 设置网格范围和步长
    x_min, y_min = 0, 0
    x_max, y_max = image_size

    # 生成网格点
    x_values = np.arange(x_min, x_max, step)
    y_values = np.arange(y_min, y_max, step)

    # 生成网格点的坐标
    grid_points = [(x, y) for x in x_values for y in y_values]

    # 创建DataFrame
    df_grid = pd.DataFrame(grid_points, columns=["x", "y"])

    # 平面中心
    assert adata.model.spot_projection.projection.in_features == num_layers, print(
        "Please check if the input dim of spot_projection == num_layers."
    )
    center_x, center_y = (x_max - x_min) / 2, (y_max - y_min) / 2

    print("Generating annotation distribution...")
    # 为每个网格点生成基于曼哈顿距离的渐变4维向量
    df_grid[f"{num_layers}d_vector"] = df_grid.apply(
        lambda row: generate_annotation_distribution(
            row["x"], row["y"], center_x, center_y, layer_size, num_layers, ord
        ),
        axis=1,
    )

    # 将4维向量展开成独立的列
    df_vectors = pd.DataFrame(
        df_grid[f"{num_layers}d_vector"].tolist(),
        columns=[f"v{i+1}" for i in range(num_layers)],
    )

    # 将网格点的坐标与4维向量合并
    df_result = pd.concat([df_grid[["x", "y"]], df_vectors], axis=1)

    annotations = df_result.iloc[:, 2:].to_numpy()
    generated_images = image_generation(adata, annotations, 1)

    # 定义图像大小
    mask = generate_mask(patch_size)

    # 生成每个点的224x224x3随机图像

    images = {
        point: generated_images[i, :, :, :].cpu() for i, point in enumerate(grid_points)
    }

    big_image_width = x_max
    big_image_height = y_max

    # 创建大图
    big_image = np.zeros((big_image_height, big_image_width, 3), dtype=np.uint8) + 255
    print("Blending image...")
    big_image = blend_images(big_image, images, patch_size, mask)

    df_tissue = df_result[
        (df_result[[f"v{i+1}" for i in range(num_layers)]] > 0).any(axis=1)
    ]
    X = np.random.randn(len(df_tissue), 2)
    adata_st = sc.AnnData(
        X=X,
    )

    # 应用坐标转移矩阵
    adata_st.obsm["spatial"] = df_tissue.iloc[:, :2].to_numpy().astype(int)

    adata_st.uns["spatial"] = {
        "test": {
            "images": {
                "hires": big_image,
            },
            "scalefactors": {
                "tissue_hires_scalef": 1,  # 根据需要调整比例因子
                "spot_diameter_fullres": step / 3 * 2,
            },
        },
    }

    adata_st.obsm["annotations"] = df_tissue.iloc[:, 2:].to_numpy()

    assert (
        "annotation_list" in adata.uns.keys()
    ), "Set annotation_list of adata! For example, [A,B,C,D] for num_layers=4."

    adata_st.uns["annotation_list"] = adata.uns["annotation_list"]

    return adata_st


def recover_tissue(
    adata,
    step=28,
    distance_threshold=160,
    k=4,
    patch_size=224,
    image_size=(10000, 10000),
    device="cuda:0",
    topk=3,
    mode = 'GFF'
):
    points = adata.obsm["spatial"]
    reference_annotations = adata.obsm["annotations"]
    x_min, y_min = 0, 0
    x_max, y_max, _ = adata.uns["spatial"]["test"]["images"]["hires"].shape
    x_ratio = x_max / image_size[0]
    y_ratio = y_max / image_size[1]
    
    # 生成网格点
    x_values = np.arange(0, image_size[0], step)
    y_values = np.arange(0, image_size[1], step)
    x_values_raw = (x_values * x_ratio).astype(int)
    y_values_raw = (y_values * y_ratio).astype(int)
    # 生成网格点的坐标
    super_points = [(x, y) for x in x_values for y in y_values]
    super_points_raw = [(x, y) for x in x_values_raw for y in y_values_raw]
    distance_threshold = 2 * (points.max() - points.min()) / len(points) ** 0.5
    print("Generating annotation distribution...")
    super_annotations = find_k_nearest_averages_with_threshold(
        super_points_raw, points, reference_annotations, k=k, threshold=distance_threshold
    )
    print("Generating patches...")
    generated_images = image_generation(adata, super_annotations, topk, device=device, mode=mode)

    # use new points on new image
    images = {
        point: generated_images[i, :, :, :].cpu()
        for i, point in enumerate(super_points)
    }
    mask = generate_mask(patch_size)

    big_image_width =  image_size[0]
    big_image_height =  image_size[1]

    # 创建大图
    big_image = np.zeros((big_image_height, big_image_width, 3), dtype=np.uint8) + 255
    print("Blending image...")
    big_image = blend_images(big_image, images, patch_size, mask)

    tissue_index = np.where(super_annotations.sum(1) > 0)
    annotation_tissue = super_annotations[tissue_index]

    X = np.random.randn(len(annotation_tissue), 2)
    adata_st = sc.AnnData(
        X=X,
    )

    # 应用坐标转移矩阵
    super_points = np.array(super_points)
    adata_st.obsm["spatial"] = super_points[tissue_index].astype(int)
    adata_st.uns["spatial"] = {
        "test": {
            "images": {
                "hires": big_image,
            },
            "scalefactors": {
                "tissue_hires_scalef": 1,  # 根据需要调整比例因子
                "spot_diameter_fullres": step / 3 * 2,
            },
        },
    }

    adata_st.obsm["annotations"] = annotation_tissue

    assert (
        "annotation_list" in adata.uns.keys()
    ), "Set annotation_list of adata! For example, [A,B,C,D] for num_layers=4."

    adata_st.uns["annotation_list"] = adata.uns["annotation_list"]

    return adata_st
