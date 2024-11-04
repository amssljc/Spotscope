import random
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import binary_fill_holes
from scipy.spatial import KDTree
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scipy.ndimage.filters import laplace
from scipy.ndimage import uniform_filter, gaussian_filter
import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def find_values_from_mult_idict(nested_dict, query_key="hires"):
    # 初始化一个列表来保存找到的所有 'hires' 值
    hires_values = []

    # 递归函数来遍历字典
    def recurse(items):
        # 遍历字典的键和值
        for key, value in items.items():
            if key == query_key:
                # 如果找到 'hires' 关键词，添加其值到列表
                hires_values.append(value)
            elif isinstance(value, dict):
                # 如果值还是一个字典，递归调用此函数
                recurse(value)

    # 开始递归遍历传入的字典
    recurse(nested_dict)
    return np.array(hires_values[0])


def setup_seed(seed=42):
    """
    设置随机种子函数，采用固定的随机种子使得结果可复现
    seed：种子值，int
    """
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)  # numpy 设置随机种子
    random.seed(seed)  # random 设置随机种子
    # torch.backends.cudnn.benchmark = False  # cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。对精度影响不大，仅仅是小数点后几位的差别，微小差异容忍可注释
    torch.backends.cudnn.deterministic = True


def uniform_points_on_simplex(K, num_points=1):
    """
    Generate uniformly distributed points inside a K-dimensional simplex (X1 + X2 + ... + XK <= 1).
    
    Parameters:
    K (int): Dimension of the simplex.
    num_points (int): Number of points to generate.
    
    Returns:
    numpy.ndarray: Array of shape (num_points, K) with the sampled points.
    """
    # Create a grid of points in the (K-1) dimensions
    divisions = int(num_points ** (1 / K)) + 1
    grid = np.linspace(0, 1, divisions)
    
    # Generate all possible combinations of points with (K-1) dimensions
    grid_points = np.array(list(itertools.product(grid, repeat=K)))
    
    # Filter points that satisfy the condition X1 + X2 + ... + XK <= 1
    valid_points = grid_points[np.sum(grid_points, axis=1) <= 1]
    
    # Sample the required number of points from the valid points uniformly
    if len(valid_points) > num_points:
        indices = np.random.choice(len(valid_points), size=num_points, replace=False)
        sampled_points = valid_points[indices]
    else:
        sampled_points = valid_points
    
    return sampled_points



def load_model(adata, model_path, model):
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("module.", "")  # remove the prefix 'module.'
        new_key = new_key.replace("well", "spot")  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    adata.model = model
    print("Finished loading model")


def get_image_embeddings(adata, name="query", device="cuda:0"):
    image_embeddings = []
    loader_name = name + "_dataloaders"
    dataloader = adata.uns[loader_name]
    model = adata.model.to(device)
    model.eval()
    print("Getting image embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_features = model.image_encoder(batch["image"].to(device))
            image_embeddings_tmp = model.image_projection(image_features)
            image_embeddings.append(image_embeddings_tmp)

    image_embeddings = torch.cat(image_embeddings).cpu().numpy()
    ebd_name = name + "_img_embeddings"
    adata.uns[ebd_name] = image_embeddings
    return image_embeddings


def get_spot_embeddings(
    adata, name="query", device="cuda:0", annotation_type="continuous"
):
    
    ebd_name = name + "_spot_embeddings"
    
    if ebd_name in adata.uns.keys():
        return adata.uns[ebd_name]
    else:
        spot_embeddings = []

        model = adata.model.to(device)
        model.eval()
        print("Getting spot embeddings...")

        with torch.no_grad():
            if annotation_type == "continuous" or name == "query":
                loader_name = name + "_dataloaders"
                if loader_name in adata.uns.keys():
                    dataloader = adata.uns[loader_name]
                    for batch in tqdm(dataloader):
                        spot_embeddings_tmp = model.spot_projection(
                            batch["annotations"].to(device)
                        )
                        spot_embeddings.append(spot_embeddings_tmp)
                    spot_embeddings = torch.cat(spot_embeddings).cpu().numpy()
                else:
                    K = len(adata.uns["annotation_list"])
                    annotations = uniform_points_on_simplex(K, min(5000, 10**K))
                    annotations = torch.as_tensor(annotations).to(device).float()
                    spot_embeddings = model.spot_projection(annotations).cpu().numpy()
            elif annotation_type == "discrete" and name == "reference":
                K = len(adata.uns["annotation_list"])
                annotations = torch.eye(K).to(device)
                spot_embeddings = model.spot_projection(annotations).cpu().numpy()

        ebd_name = name + "_spot_embeddings"
        adata.uns[ebd_name] = spot_embeddings
        return spot_embeddings


def get_query_coordinates(adata):
    coordinates = []
    loader_name = "query" + "_dataloaders"
    dataloader = adata.uns[loader_name]
    print("Getting query coordinates...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            coordinates.append(batch["relative_coords"])

    coordinates = torch.cat(coordinates).cpu().numpy()
    adata.uns["spot_coordinates"] = coordinates
    return coordinates


def get_reference_annotations(adata, annotation_type="continuous"):
    annotations = []
    if annotation_type == "continuous":
        loader_name = "reference" + "_dataloaders"
        if loader_name in adata.uns.keys():
            dataloader = adata.uns[loader_name]
            print("Getting reference annotations...")
            for batch in tqdm(dataloader):
                annotations.append(batch["annotations"])
            annotations = np.concatenate(annotations, axis=0)
        else:
            K = len(adata.uns["annotation_list"])
            annotations = uniform_points_on_simplex(K, min(5000, 10**K))
    elif annotation_type == "discrete":
        K = len(adata.uns["annotation_list"])
        annotations = np.eye(K)
    adata.uns["reference_annotations"] = annotations
    return annotations


def find_matches(reference_embeddings, query_embeddings, topk=1):
    # find the closest matches
    reference_embeddings = torch.tensor(reference_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    reference_embeddings = F.normalize(reference_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ reference_embeddings.T  # 2277x2265
    print("dot_similarity shape:", dot_similarity.shape)
    topsimilarity, indices = torch.topk(dot_similarity.squeeze(0), k=topk)

    return topsimilarity.cpu().numpy(), indices.cpu().numpy()


def normalize_celltype(arr, perc):
    # 计算百分位数
    lower = np.percentile(
        arr, perc * 100, axis=0, keepdims=True
    )  # keepdims保持维度对齐

    # 应用clip操作
    arr_clipped = np.clip(arr, lower, arr.max())
    # 计算每列的最小值和最大值
    min_vals = np.min(arr_clipped, axis=0, keepdims=True)
    max_vals = np.max(arr_clipped, axis=0, keepdims=True)

    # 进行归一化
    arr_normalized = (arr_clipped - min_vals) / (max_vals - min_vals + 1e-10)

    return arr_normalized


def softmax(x):
    # 减去每列的最大值以提高数值稳定性
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def calculate_similarity(true, pred, method="pcc", wise="spot"):
    if wise == "spot":
        # similarity between spots
        axis = 0
    elif wise == "feature":
        # similarity between celltype
        axis = 1

    corrs = np.zeros(pred.shape[axis])
    for i in range(pred.shape[axis]):
        if axis == 0:
            pred_ = pred[i, :]
            true_ = true[i, :]
        elif axis == 1:
            pred_ = pred[:, i]
            true_ = true[:, i]

        if method == "pcc":
            corr = np.corrcoef(
                pred_,
                true_,
            )[0, 1]

        if np.isnan(corr):
            corrs[i] = 0
        else:
            corrs[i] = corr

    print(f"Mean correlation across {wise}: ", np.mean(corrs))
    return corrs


def plot_annotations(
    adata,
    annotation_reference="inferred_spot_annotations",
    dpi_save=300,
    save_path=None,
    alpha_img=0.7,
    font_size=10,
):
    plt.rcParams.update({"font.size": font_size})
    celltypes = adata.uns["annotation_list"]
    for i, ct in enumerate(celltypes):
        adata.obs[ct] = adata.obsm[annotation_reference][:, i]
    if save_path:
        sc.pl.spatial(
            adata,
            color=celltypes,
            show=False,
            alpha_img=alpha_img,
        )
        plt.savefig(save_path, dpi=dpi_save)
    else:
        sc.pl.spatial(adata, color=celltypes, alpha_img=alpha_img)


def filter_points(A, B, pixel_buffer=0):
    # 获取A的最小和最大坐标
    min_x, min_y = np.min(A, axis=0)
    max_x, max_y = np.max(A, axis=0)

    # 计算扩展边界
    min_x_ext, min_y_ext = min_x - pixel_buffer, min_y - pixel_buffer
    max_x_ext, max_y_ext = max_x + pixel_buffer, max_y + pixel_buffer

    # 过滤B中的点
    B_filtered = []
    for point in B:
        x, y = point
        if min_x_ext <= x <= max_x_ext and min_y_ext <= y <= max_y_ext:
            B_filtered.append(point)

    return np.array(B_filtered)


def generate_super_points(points, size, scale, distance_threshold):
    # 创建一个更密集的网格，这次仅关注坐标生成
    x = np.linspace(min(points[:, 0]), max(points[:, 0]), size * scale)
    y = np.linspace(min(points[:, 1]), max(points[:, 1]), size * scale)
    grid_x, grid_y = np.meshgrid(x, y)

    # 将生成的网格坐标扁平化，以便进行插值
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()

    # 使用griddata插值，仅插值坐标
    interpolated_points = griddata(
        points, points, (grid_x_flat, grid_y_flat), method="linear"
    )

    # 清除所有NaN值（插值未定义的区域）
    valid_points = interpolated_points[~np.isnan(interpolated_points).any(axis=1)]

    # 使用KDTree找到每个插值点到最近原始点的距离
    tree = KDTree(points)
    distances, _ = tree.query(valid_points, k=1, p=100)  # 查找最近的一个原始点

    # 过滤掉超过阈值距离的插值点
    filtered_points = valid_points[distances < distance_threshold]

    super_points = filtered_points.astype(int)
    # 假设已经有新的坐标数组new_arr，它是一个(N_new, 2)的NumPy数组

    super_points = filter_points(points, super_points, 0)
    return super_points


def is_in_tissue_area(x, y, mask, size, threshold):
    size = max(size, 112)
    x_min, x_max = max(x - size, 0), min(x + size, mask.shape[1])
    y_min, y_max = max(y - size, 0), min(y + size, mask.shape[0])

    sub_mask = mask[y_min:y_max, x_min:x_max]
    density = np.mean(sub_mask) / 255
    return density > threshold


def process_image_to_adata(
    image_path,
    grid_size=64,
    density_threshold=0.9,
    lower_range=[0, 30, 50],
    upper_range=[180, 255, 255],
    patch_size=224,
):
    print("Detecting coordinates in tissue...")
    image = Image.open(image_path)
    image_array = np.array(image)
    image_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    lower_range = np.array(lower_range)
    upper_range = np.array(upper_range)
    # Create masks for the regions
    mask = cv2.inRange(image_hsv, lower_range, upper_range)

    # Apply morphological operations to clean up the masks
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_solid = binary_fill_holes(mask_clean).astype(np.uint8) * 255

    # Create a grid of points
    rows, cols = mask_solid.shape

    # Calculate the tissue area for each grid point
    points = []
    for y in range(0, rows, grid_size):
        for x in range(0, cols, grid_size):
            in_tissue = is_in_tissue_area(
                x, y, mask_solid, grid_size, density_threshold
            )
            if (
                (x - patch_size / 2 > 0)
                and (y - patch_size / 2 > 0)
                and (x + patch_size / 2 <= image_array.shape[1])
                and (y + patch_size / 2 <= image_array.shape[0])
            ):
                points.append([x, y, in_tissue])

    # Create a DataFrame
    df = pd.DataFrame(points, columns=["x", "y", "in_tissue"])
    df_tissue = df[df["in_tissue"]]

    X = np.random.randn(len(df_tissue), 2)
    adata_st = sc.AnnData(
        X=X,
    )

    # 应用坐标转移矩阵
    adata_st.obsm["spatial"] = df_tissue.iloc[:, :2].to_numpy().astype(int)

    adata_st.uns["spatial"] = {
        "test": {
            "images": {
                "hires": image_array,
            },
            "scalefactors": {
                "tissue_hires_scalef": 1,  # 根据需要调整比例因子
                "spot_diameter_fullres": grid_size / 3 * 2,
            },
        },
    }

    return adata_st


def gaussian(x, mu, sigma):
    return np.exp(-np.power((x - mu) / sigma, 2.0) / 2.0)


def generate_annotation_distribution(
    x, y, center_x, center_y, layer_size, num_layers, ord=2
):
    point = np.array([x, y])
    center = np.array([center_x, center_y])
    distance = np.linalg.norm(point - center, ord=ord)

    sigma = layer_size / 2.0  # Standard deviation, can be adjusted as needed

    if distance > num_layers * layer_size:
        return [0] * num_layers

    annotations = []
    for i in range(num_layers):
        annotations.append(gaussian(distance, i * layer_size, sigma))

    return annotations


def pyramid_blending(A, B, mask):
    # 确保图像是float32类型以便处理
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    mask = mask.astype(np.float32)

    # 生成高斯金字塔
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    G = mask.copy()
    gpM = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpM.append(G)

    # 生成拉普拉斯金字塔
    lpA = [gpA[-1]]
    for i in range(6, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        GE = cv2.resize(GE, (gpA[i - 1].shape[1], gpA[i - 1].shape[0]))
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    lpB = [gpB[-1]]
    for i in range(6, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        GE = cv2.resize(GE, (gpB[i - 1].shape[1], gpB[i - 1].shape[0]))
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    gpM.reverse()
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpM):
        gm = cv2.resize(gm, (la.shape[1], la.shape[0]))
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # 重建图像
    ls_ = LS[0]
    for i in range(1, len(LS)):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i])

    # 确保输出在0-255并转换为uint8类型
    ls_ = np.clip(ls_, 0, 255)
    ls_ = ls_.astype(np.uint8)

    return ls_


def blend_images(big_image, images, patch_size, mask):
    for (x, y), img in tqdm(images.items()):
        start_x = x
        start_y = y
        try:
            # 提取大图和小图的对应区域
            region_big_image = big_image[
                start_y : start_y + patch_size, start_x: start_x + patch_size
            ].astype(np.float32)
            region_img = img.numpy().astype(np.float32)
            # 金字塔融合
            blended_region = pyramid_blending(region_big_image, region_img, mask)

            # 将融合后的区域放回大图
            big_image[
                start_y : start_y + patch_size, start_x : start_x + patch_size
            ] = blended_region
        except Exception as e:  # noqa: F841
            # print(f"Error blending image at position ({x}, {y}): {e}")
            continue

    return big_image


def str2array(arr_str):
    # 去除括号和换行符并将字符串分割为单个数字
    array_str_cleaned = arr_str.replace("[", "").replace("]", "").replace("\n", " ")
    array_list = [float(x) for x in array_str_cleaned.split()]

    # 将列表转换为NumPy数组
    array = np.array(array_list)
    return array


def calculate_coord_similarity(coordinates, gamma=1.0):
    """
    计算给定坐标点之间的相似性。

    参数:
    - coordinates: 可以是 numpy.ndarray 或 torch.Tensor，形状为(N, D)
    - gamma: 相似度计算中使用的衰减因子

    返回:
    - similarity: 相似性矩阵，形状为(N, N)，类型为 torch.Tensor
    """
    if isinstance(coordinates, np.ndarray):
        # 处理 numpy.ndarray 输入
        coordinates = torch.from_numpy(coordinates)
    elif not isinstance(coordinates, torch.Tensor):
        raise TypeError("输入必须是 numpy.ndarray 或 torch.Tensor")

    # 计算距离矩阵
    dists = torch.sqrt(
        torch.sum((coordinates[:, None, :] - coordinates[None, :, :]) ** 2, dim=2)
    )

    # 转换距离到相似度
    similarity = torch.exp(-gamma * dists**2)

    return similarity


def normalize_tensor_to_neg1_1(X):
    """
    将张量 X 的元素归一化到 [-1, 1] 范围。

    参数:
    - X (torch.Tensor): 需要归一化的张量。

    返回:
    - torch.Tensor: 归一化后的张量。
    """
    X_min = torch.min(X)
    X_max = torch.max(X)
    X_normalized = 2 * ((X - X_min) / (X_max - X_min)) - 1
    return X_normalized


def find_k_nearest_averages_with_threshold(
    N_points, M_references, annotations, k=4, threshold=150
):
    """
    对于给定的N个点，找到M个参考点中最近的k个点，然后计算这些点的注释信息的平均值。
    如果前k个距离中的任何一个超过指定阈值，返回0向量。

    参数:
    N_points : numpy.ndarray
        N x 2 数组，表示N个点的坐标。
    M_references : numpy.ndarray
        M x 2 数组，表示M个参考点的坐标。
    annotations : numpy.ndarray
        M x D 数组，表示每个参考点的注释信息。
    k : int
        指定需要找到的最近参考点的数量。
    threshold : float
        距离阈值，超过这个阈值的点的注释将被设置为0向量。

    返回:
    numpy.ndarray
        N x D 数组，每行是对应N点的注释信息的平均值或0向量。
    """
    # 计算距离
    N_points = np.array(N_points)
    M_references = np.array(M_references)
    annotations = np.array(annotations)

    distances = np.linalg.norm(
        N_points[:, np.newaxis, :] - M_references[np.newaxis, :, :], axis=2
    )

    # 找到每个N点对应的最近k个M点的索引和距离
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_distances = np.sort(distances, axis=1)[:, :k]

    # 初始化结果数组
    average_annotations = np.zeros((N_points.shape[0], annotations.shape[1]))

    # 计算注释信息的平均值或返回0向量
    for i in range(N_points.shape[0]):
        if np.all(nearest_distances[i] <= threshold):
            average_annotations[i] = np.mean(annotations[nearest_indices[i]], axis=0)
        else:
            average_annotations[i] = np.zeros(annotations.shape[1])

    return average_annotations


def generate_mask(patch_size):
    # 定义图像大小
    rows, cols = patch_size, patch_size  # 你可以根据实际图像大小调整

    # 创建一个全1的掩码
    mask = np.ones((rows, cols, 3), dtype=np.float32)

    # 定义中心正方形的大小
    square_size = int(patch_size / 1.5)
    center_row, center_col = rows // 2, cols // 2

    # 设置中心正方形的值为0
    mask[
        center_row - square_size // 2 : center_row + square_size // 2,
        center_col - square_size // 2 : center_col + square_size // 2,
    ] = 0

    # 应用高斯模糊来创建平滑的渐变效果
    mask = cv2.GaussianBlur(mask, (127, 127), 0)
    return mask


def clustering_metrics(
    adata, predict_obs="embedding", target_obs="domain_annotation", verbose=True
):
    assert target_obs in adata.obs.keys(), print("domain_annotation is not availiable!")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import pandas as pd

    df = pd.DataFrame(
        {"predict": adata.obs[predict_obs], "ground_truth": adata.obs[target_obs]}
    )
    df = df.dropna()
    ari = adjusted_rand_score(
        df["ground_truth"],
        df["predict"],
    )
    nmi = normalized_mutual_info_score(
        df["ground_truth"],
        df["predict"],
    )
    adata.uns[f"{predict_obs}_ARI"] = ari
    adata.uns[f"{predict_obs}_NMI"] = nmi
    if verbose:
        print(f"ARI of {predict_obs} is: {ari}")
        print(f"NMI of {predict_obs} is: {nmi}")

    return ari, nmi


def evaluate_performance(predictions, onehot_labels, mode="all-class"):
    """
    评估预测结果与one-hot标签的性能。

    :param predictions: N x D 维度的预测值矩阵
    :param onehot_labels: N x D 维度的one-hot标签矩阵
    :param mode: 评估模式，'A' 为全面评估，'B' 为二分类评估
    :return: 一个包含各种性能指标的字典
    """
    # 计算预测标签
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(onehot_labels, axis=1)

    if mode == "2-class":
        # 将前D-1列合并为一个标签进行二分类评估
        predicted_labels = np.where(
            predicted_labels == (onehot_labels.shape[1] - 1), 1, 0
        )
        true_labels = np.where(true_labels == (onehot_labels.shape[1] - 1), 1, 0)

    # 检查标签是否都是相同的
    if np.all(true_labels == true_labels[0]):
        print("All labels are the same. Skipping detailed metric calculations.")
        return {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "ari": "N/A",
            "nmi": "N/A",
        }

    # 准确率
    accuracy = accuracy_score(true_labels, predicted_labels)

    # 精确率
    precision = precision_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )

    # 召回率
    recall = recall_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    )

    # F1分数
    f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

    # 调整兰德指数（ARI）
    ari = adjusted_rand_score(true_labels, predicted_labels)

    # 归一化互信息（NMI）
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    # 返回所有指标
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "ari": ari,
        "nmi": nmi,
    }

    return metrics



# Algorithm using scipy's guided Filter

def fusion(im1,im2):
    sigma_r = 5
    average_filter_size=31
    r_1=45
    r_2=7
    eps_1=0.3
    eps_2=10e-6
    

    if im1.max()>1:
        im1=im1/255
    if im2.max()>1:
        im2=im2/255
            
    im1_blue, im1_green, im1_red = cv2.split(im1)
    im2_blue, im2_green, im2_red = cv2.split(im2)
            
    base_layer1 = uniform_filter(im1, mode='reflect',size=average_filter_size)
    b1_blue, b1_green, b1_red = cv2.split(base_layer1)
            
    base_layer2 = uniform_filter(im2, mode='reflect',size=average_filter_size)
    b2_blue, b2_green, b2_red = cv2.split(base_layer2)
            
    detail_layer1 = im1 - base_layer1
    d1_blue, d1_green, d1_red = cv2.split(detail_layer1)
            
    detail_layer2 = im2 - base_layer2
    d2_blue, d2_green, d2_red = cv2.split(detail_layer2)
            
            
    saliency1 = gaussian_filter(abs(laplace(im1_red+im1_green+im1_blue,mode='reflect')),sigma_r,mode='reflect')
    saliency2 = gaussian_filter(abs(laplace(im2_red+im2_green+im2_blue,mode='reflect')),sigma_r,mode='reflect')
    mask = np.float32(np.argmax([saliency1, saliency2], axis=0))
    
    im1=np.float32(im1)
    im2=np.float32(im2)
    
    gf1 = cv2.ximgproc.createGuidedFilter(im1, r_1, eps_1)
    gf2 = cv2.ximgproc.createGuidedFilter(im2, r_1, eps_1)  
    gf3 = cv2.ximgproc.createGuidedFilter(im1, r_2, eps_2)
    gf4 = cv2.ximgproc.createGuidedFilter(im2, r_2, eps_2)
        
    g1r1 = gf1.filter(1 - mask)
    g2r1 = gf2.filter(mask)
    g1r2 = gf3.filter(1-mask)
    g2r2 = gf4.filter(mask)
    
    fused_base1 = np.float32((b1_blue * (g1r1) + b2_blue * (g2r1))/((g1r1+g2r1)))          
    fused_detail1 = np.float32((d1_blue * (g1r2) + d2_blue * (g2r2))/((g1r2+g2r2)))  
    fused_base2 = np.float32((b1_green * g1r1 + b2_green * g2r1)/((g1r1+g2r1)))   
    fused_detail2 = np.float32((d1_green * (g1r2) + d2_green * (g2r2))/((g1r2+g2r2)))    
    fused_base3 = np.float32((b1_red * (g1r1) + b2_red * (g2r1))/((g1r1+g2r1)))
    fused_detail3 = np.float32((d1_red * (g1r2) + d2_red * (g2r2))/((g1r2+g2r2)))
        
        
    B1=np.float32(fused_base1+fused_detail1)
    B2=np.float32(fused_base2+fused_detail2)
    B3=np.float32(fused_base3+fused_detail3)
    
    fusion1=np.float32(cv2.merge((B1, B2, B3)))
    fusion1=fusion1/fusion1.max()
    fusion1 = (fusion1*255).astype(int)
    fusion1 = np.clip(fusion1, 0, 255)
    return fusion1