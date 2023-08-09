import cv2
import os

video_path = 'G:/Image_Decomposition/nir-main-Ran/data/result_720_scene.mp4'
output_folder = 'G:/Image_Decomposition/nir-main-Ran/data/frames'  # 抽取的帧将保存在此文件夹中

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频文件是否成功打开
if not cap.isOpened():
    print(f"错误：无法打开位于 {video_path} 的视频文件")
    exit()

frame_count = 0

# 循环遍历每一帧并将每一帧保存为 PNG 图片
while True:
    # 读取视频的一帧
    ret, frame = cap.read()

    # 如果视频结束，则退出循环
    if not ret:
        break

    # 将帧保存为 PNG 图片
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# 释放视频捕获对象并关闭视频文件
cap.release()

print(f"抽取的帧数：{frame_count}")
