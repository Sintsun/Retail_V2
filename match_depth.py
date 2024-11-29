import pyrealsense2 as rs
import numpy as np
import cv2
import os

def draw_axes(img, origin, imgpts):
    imgpts = imgpts.astype(int)
    print(f"原点坐标 (图像坐标系): {origin}, 类型: {type(origin)}")
    for i, pt in enumerate(imgpts):
        pt_tuple = tuple(pt.ravel())
        print(f"坐标轴 {i} 点 (图像坐标系): {pt_tuple}, 类型: {type(pt_tuple)}")
        # 定义坐标轴颜色：X - 红色, Y - 绿色, Z - 蓝色
        color = (0, 0, 255) if i == 0 else (0, 255, 0) if i == 1 else (255, 0, 0)
        img = cv2.line(img, origin, pt_tuple, color, 3)
    return img

def load_vectors(rvec_path, tvec_path):
    if not os.path.exists(rvec_path):
        raise FileNotFoundError(f"旋转向量文件未找到: {rvec_path}")
    if not os.path.exists(tvec_path):
        raise FileNotFoundError(f"位移向量文件未找到: {tvec_path}")

    rvec = np.load(rvec_path)
    tvec = np.load(tvec_path)

    print(f"加载的 rvec 形状: {rvec.shape}")
    print(f"加载的 tvec 形状: {tvec.shape}")

    # 压缩维度
    rvec = rvec.squeeze()
    tvec = tvec.squeeze()

    # 确保形状为 (3, 1)
    if rvec.shape == (1, 3):
        rvec = rvec.T
    elif rvec.shape == (3,):
        rvec = rvec.reshape(3, 1)
    elif rvec.shape != (3, 1):
        raise ValueError(f"旋转向量必须是形状 (3,1) 或 (1,3)，但得到 {rvec.shape}")

    if tvec.shape == (1, 3):
        tvec = tvec.T
    elif tvec.shape == (3,):
        tvec = tvec.reshape(3, 1)
    elif tvec.shape != (3, 1):
        raise ValueError(f"位移向量必须是形状 (3,1) 或 (1,3)，但得到 {tvec.shape}")

    print(f"最终 rvec 形状: {rvec.shape}")
    print(f"最终 tvec 形状: {tvec.shape}")
    print(f"旋转向量 (rvec):\n{rvec}")
    print(f"位移向量 (tvec):\n{tvec}")

    return rvec.astype(np.float32), tvec.astype(np.float32)

def main():
    image_folder = r"./data"  # 替换为您的图像文件夹路径
    rvec_path = os.path.join(image_folder, "rotation_vectors.npy")
    tvec_path = os.path.join(image_folder, "translation_vectors.npy")
    camera_matrix_path = os.path.join(image_folder, "camera_matrix.npy")

    # 加载相机内参矩阵
    if not os.path.exists(camera_matrix_path):
        raise FileNotFoundError(f"内参矩阵文件未找到: {camera_matrix_path}")
    camera_matrix = np.load(camera_matrix_path).astype(np.float32)
    print("相机内参矩阵 (camera_matrix):\n", camera_matrix)

    # 假设无畸变
    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    # 加载旋转和位移向量
    try:
        rvec, tvec = load_vectors(rvec_path, tvec_path)
        print("旋转向量 (rvec):\n", rvec)
        print("位移向量 (tvec):\n", tvec)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        input("按任意键退出...")
        return

    # 初始化 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用深度和颜色流
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 启动管道
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("无法启动 RealSense 管道:", e)
        input("按任意键退出...")
        return

    # 创建对齐对象，将深度帧对齐到颜色帧
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 初始化点云（如果需要）
    pc = rs.pointcloud()

    # 定义坐标轴长度（以毫米为单位，例如50毫米表示5厘米）
    axis_length = 150  # 50毫米
    axes_3d = np.float32([
        [axis_length, 0, 0],  # X-axis (Red)
        [0, axis_length, 0],  # Y-axis (Green)
        [0, 0, axis_length]  # Z-axis (Blue)
    ])

    try:
        while True:
            # 等待帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 生成点云并映射到颜色帧
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # 将图像转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 投影坐标轴点
            imgpts, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.astype(int).reshape(-1, 2)

            # 投影原点
            origin_world = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            origin_img, _ = cv2.projectPoints(origin_world, rvec, tvec, camera_matrix, dist_coeffs)
            origin_img = tuple(origin_img[0].ravel().astype(int))

            # 调试信息
            print(f"投影原点: {origin_img}")
            for i, pt in enumerate(imgpts):
                print(f"投影坐标轴 {i} 点: {tuple(pt)}")

            # 获取图像尺寸
            img_height, img_width = color_image.shape[:2]

            # 检查原点是否在图像内
            if not (0 <= origin_img[0] < img_width and 0 <= origin_img[1] < img_height):
                print("原点不在图像范围内")

            # 检查坐标轴点是否在图像内
            for i, pt in enumerate(imgpts):
                if not (0 <= pt[0] < img_width and 0 <= pt[1] < img_height):
                    print(f"坐标轴 {i} 点 {tuple(pt)} 不在图像范围内")

            # 绘制坐标轴
            img_with_axes = draw_axes(color_image.copy(), origin_img, imgpts)

            # 标记原点
            cv2.circle(img_with_axes, origin_img, 5, (0, 0, 255), -1)  # 红色圆点

            # 显示图像
            cv2.imshow('RealSense - World Axes', img_with_axes)
            cv2.imshow('Depth Image', depth_image)

            # 退出条件：按 'q' 或 'ESC' 键
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                print("退出键被按下。正在退出...")
                break

    except KeyboardInterrupt:
        print("程序被用户中断。")
    except Exception as e:
        print("发生错误:", e)
        input("按任意键退出...")
    finally:
        # 停止管道并关闭窗口
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()