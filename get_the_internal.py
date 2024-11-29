import pyrealsense2 as rs
import numpy as np

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取颜色帧的内参
color_frame = pipeline.wait_for_frames().get_color_frame()
intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

print("从 RealSense 获取的相机内参矩阵:\n", camera_matrix)

# 停止管道
pipeline.stop()