import numpy as np

def kmeans_manual(data, k, max_iters=100, tol=1e-4):
    """
    手动实现 K-Means 算法
    :param data: (N, D) 形状的 numpy 数组，每行是一个 D 维数据点
    :param k: 聚类个数
    :param max_iters: 最大迭代次数
    :param tol: 质心变化的阈值
    :return: (k, D) 形状的锚框中心
    """
    # 1️⃣ 随机初始化 k 个中心点（从数据中随机选 k 个点）
    np.random.seed(42)
    centroids = data[np.random.choice(len(data), k, replace=False)]

    for _ in range(max_iters):
        # 2️⃣ 计算每个数据点到 `k` 个质心的距离（欧几里得距离）
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)  # (N, k)
        
        # 3️⃣ 归类：每个数据点归到最近的质心
        labels = np.argmin(distances, axis=1)  # (N,)

        # 4️⃣ 计算新的质心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # 5️⃣ 判断是否收敛（质心变化很小）
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids

    return centroids
