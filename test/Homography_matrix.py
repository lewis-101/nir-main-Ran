import torch
import torch.nn as nn
import numpy as np
import cv2


class HomographyMLP(nn.Module):
    def __init__(self, in_features=4, hidden_features=256, hidden_layers=1):
        super().__init__()
        out_features = 8

        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

        self.init_weights()

    def forward(self, x):
        return self.net(x)


# 读取两张图片
image1 = cv2.imread("G:/Image_Decomposition/nir-main-Ran/test/picture/1.jpg")
image2 = cv2.imread("G:/Image_Decomposition/nir-main-Ran/test/picture/2.jpg")

# 创建特征点检测器
detector = cv2.ORB_create()

# 在两张图片上检测特征点
keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

# 创建特征点匹配器
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 对特征点进行匹配
matches = matcher.match(descriptors1, descriptors2)

# 提取匹配点对应的特征点坐标
image1_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
image2_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

# 构建 MLP 模型
mlp = HomographyMLP()

# 将特征点坐标拼接为输入数据，形状为 (N, 4)
model_input = torch.cat((torch.from_numpy(image1_points), torch.from_numpy(image2_points)), dim=1)

# 计算单应性矩阵
homography_matrix = mlp(model_input)

# 输出单应性矩阵
print("Homography matrix:")
print(homography_matrix)
