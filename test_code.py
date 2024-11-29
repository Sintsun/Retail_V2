import pyrealsense2 as rs
import numpy as np
import cv2
import os
import open3d as o3d
import threading

print(f"Open3D Version: {o3d.__version__}")


def draw_axes(img, origin, imgpts):
    """
    在图像上绘制3D坐标轴。

    :param img: 要绘制的图像。
    :param origin: 原点的图像坐标。
    :param imgpts: 投影后的轴点的图像坐标。
    :return: 绘制坐标轴后的图像。
    """
    imgpts = imgpts.astype(int)
    for i, pt in enumerate(imgpts):
        pt_tuple = tuple(pt.ravel())
        # 定义轴的颜色：X - 红色, Y - 绿色, Z - 蓝色
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i]
        img = cv2.line(img, origin, pt_tuple, color, 3)
    return img


def load_vectors(rvec_path, tvec_path):
    """
    从.npy文件中加载旋转向量和平移向量。

    :param rvec_path: 旋转向量文件路径。
    :param tvec_path: 平移向量文件路径。
    :return: 旋转向量（rvec）和平移向量（tvec）的Float32类型numpy数组，形状为(3,1)。
    """
    if not os.path.exists(rvec_path):
        raise FileNotFoundError(f"未找到旋转向量文件: {rvec_path}")
    if not os.path.exists(tvec_path):
        raise FileNotFoundError(f"未找到平移向量文件: {tvec_path}")

    rvec = np.load(rvec_path).astype(np.float32).squeeze()
    tvec = np.load(tvec_path).astype(np.float32).squeeze()

    # 如果需要，调整为(3,1)形状
    rvec = rvec.reshape(3, 1) if rvec.shape == (3,) else rvec
    tvec = tvec.reshape(3, 1) if tvec.shape == (3,) else tvec

    print(f"加载的rvec形状: {rvec.shape}")
    print(f"加载的tvec形状: {tvec.shape}")

    return rvec.astype(np.float32), tvec.astype(np.float32)


def load_distortion_coeffs(dist_coeffs_path):
    """
    从.npy文件中加载畸变系数。如果文件不存在，则假设没有畸变。

    :param dist_coeffs_path:畸变系数文件路径。
    :return: 畸变系数的Float32类型numpy数组。
    """
    if not os.path.exists(dist_coeffs_path):
        print(f"未找到畸变系数文件: {dist_coeffs_path}. 假设没有畸变。")
        return np.zeros((5, 1), dtype=np.float32)  # 通常，畸变有5个系数
    dist_coeffs = np.load(dist_coeffs_path).astype(np.float32)
    print("畸变系数已加载。")
    return dist_coeffs


def initialize_realsense():
    """
    初始化RealSense管道，启用深度和RGB流，获取相机内参和畸变系数。

    :return: RealSense管道对象、深度缩放单位、相机内参矩阵、畸变系数、点云对象。
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用深度和颜色流，分辨率1280x720，帧率30
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 启动管道
    profile = pipeline.start(config)

    # 获取深度传感器的深度缩放单位
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度缩放单位: {depth_scale} 米/单位")

    # 获取RGB相机内参
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    # 创建相机内参矩阵
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    print("相机内参矩阵:\n", camera_matrix)

    # 获取畸变系数
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)
    print("畸变系数:\n", dist_coeffs)

    # 初始化点云对象
    pointcloud = rs.pointcloud()

    return pipeline, depth_scale, camera_matrix, dist_coeffs, pointcloud


def load_transformation_data(output_folder="./data"):
    """
    加载保存的旋转向量、平移向量和相机内参矩阵。

    :param output_folder: 输出文件夹路径
    :return: tuple 包含 rotation_vector, translation_vector, rgb_intrinsic_matrix, dist_coeffs
    """
    rvec_path = os.path.join(output_folder, "rotation_vectors.npy")
    tvec_path = os.path.join(output_folder, "translation_vectors.npy")
    camera_matrix_path = os.path.join(output_folder, "rgb_intrinsic_matrix.npy")
    dist_coeffs_path = os.path.join(output_folder, "distortion_coeffs.npy")  # 畸变系数路径

    # 加载相机内参矩阵
    if not os.path.exists(camera_matrix_path):
        raise FileNotFoundError(f"未找到相机内参矩阵文件: {camera_matrix_path}")
    camera_matrix = np.load(camera_matrix_path).astype(np.float32)
    print("相机内参矩阵 (camera_matrix):\n", camera_matrix)

    # 加载畸变系数
    dist_coeffs = load_distortion_coeffs(dist_coeffs_path)

    # 加载旋转向量和平移向量
    try:
        rvec, tvec = load_vectors(rvec_path, tvec_path)
        print("旋转向量 (rvec):\n", rvec)
        print("平移向量 (tvec):\n", tvec)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        input("按任意键退出...")
        return None, None, None, None

    return rvec, tvec, camera_matrix, dist_coeffs


def copy_pointcloud(pcd):
    """
    手动创建一个Open3D PointCloud对象的深拷贝。

    :param pcd: 原始PointCloud对象。
    :return: 新的PointCloud对象，包含相同的数据。
    """
    pcd_copy = o3d.geometry.PointCloud()
    pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))

    if pcd.has_colors():
        pcd_copy.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))

    if pcd.has_normals():
        pcd_copy.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))

    return pcd_copy


def apply_transformation_to_pointcloud(pcd, rvec, tvec):
    """
    将旋转向量和平移向量应用到点云。

    :param pcd: Open3D点云对象
    :param rvec: 旋转向量
    :param tvec: 平移向量
    :return: 变换后的点云
    """
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation_vector = tvec.flatten()

    # 构建4x4变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    # 应用变换
    pcd.transform(transformation_matrix)
    return pcd


def main():
    # 加载变换数据
    output_folder = "./data"
    rvec, tvec, camera_matrix, dist_coeffs = load_transformation_data(output_folder)
    if rvec is None or tvec is None:
        print("变换数据加载失败，程序退出。")
        return

    # 初始化RealSense管道
    pipeline, depth_scale, camera_matrix_realsense, distortion_coeffs, pointcloud = initialize_realsense()

    # 如果加载的相机内参与RealSense的内参不同，决定使用哪个
    if not np.allclose(camera_matrix, camera_matrix_realsense):
        print("警告：加载的相机内参与RealSense的内参不一致。使用RealSense内参。")
        camera_matrix = camera_matrix_realsense
        dist_coeffs = distortion_coeffs

    # 初始化Open3D可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='实时点云变换', width=1280, height=720)
    pcd_o3d = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_o3d)

    # 用于线程安全的锁
    lock = threading.Lock()

    # 定义鼠标点击回调函数（用于采样）
    clicked_points_world = []
    last_clicked_position = None
    last_clicked_world = None

    # 因为OpenCV的回调函数需要访问depth_frame，我们需要一个共享变量
    global_depth_frame = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points_world, last_clicked_position, last_clicked_world, global_depth_frame
        if event == cv2.EVENT_LBUTTONDOWN and global_depth_frame is not None:
            print(f"鼠标点击位置: ({x}, {y})")
            # 获取深度值
            depth = global_depth_frame.get_distance(x, y) * depth_scale
            if depth == 0:
                print("无效的深度值。")
                return
            # 获取3D坐标（相机坐标系）
            X = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
            Y = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
            Z = depth
            point_cam = np.array([[X], [Y], [Z]], dtype=np.float32)
            # 转换到世界坐标系
            R, _ = cv2.Rodrigues(rvec)
            R_inv = R.T  # 因为 R 是正交矩阵，逆矩阵等于转置矩阵
            point_world = R_inv @ (point_cam - tvec)

            print(f"相机坐标系下的3D点: X={X:.4f} Y={Y:.4f} Z={Z:.4f} 米")
            print(
                f"世界坐标系下的3D点: X={point_world[0][0]:.4f} Y={point_world[1][0]:.4f} Z={point_world[2][0]:.4f} 米")
            clicked_points_world.append(point_world.flatten())

            last_clicked_position = (x, y)
            last_clicked_world = point_world.flatten()

    # 创建OpenCV窗口并设置鼠标回调
    cv2.namedWindow('RealSense - World Axes')
    cv2.setMouseCallback('RealSense - World Axes', mouse_callback)

    try:
        while True:
            # 等待获取帧
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 将点云映射到RGB帧并计算点云
            pointcloud.map_to(color_frame)
            points = pointcloud.calculate(depth_frame)

            # 将当前深度帧设置为全局变量，以供回调函数使用
            global_depth_frame = depth_frame

            # 获取顶点并重塑为(height, width, 3)
            try:
                # 使用np.asarray正确解释缓冲区
                vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(
                    (depth_frame.height, depth_frame.width, 3))
            except ValueError as ve:
                print(f"发生错误: {ve}")
                print("请检查点云数据是否正确提取。")
                input("按任意键退出...")
                break
            except AttributeError as ae:
                print(f"发生错误: {ae}")
                print("请确保pyrealsense2已正确安装和导入。")
                input("按任意键退出...")
                break

            # 检查是否有有效的点
            if vertices.size == 0:
                print("捕获的点云为空，跳过此帧。")
                continue

            # 创建Open3D点云对象
            pcd_current = o3d.geometry.PointCloud()
            points_o3d = vertices.reshape(-1, 3)
            pcd_current.points = o3d.utility.Vector3dVector(points_o3d)

            # 检查点云是否为空
            if not pcd_current.has_points():
                print("当前点云没有点，跳过此帧。")
                continue

            # 应用变换
            with lock:
                try:
                    # 尝试使用 clone 方法
                    pcd_transformed = apply_transformation_to_pointcloud(pcd_current.clone(), rvec, tvec)
                except AttributeError:
                    # 如果 'clone' 方法不存在，使用自定义的 copy_pointcloud 函数复制点云
                    print("'clone' 方法不存在，使用自定义的 copy_pointcloud 函数复制点云。")
                    pcd_copy = copy_pointcloud(pcd_current)
                    pcd_transformed = apply_transformation_to_pointcloud(pcd_copy, rvec, tvec)

                # 更新Open3D可视化
                pcd_o3d.points = pcd_transformed.points
                if pcd_transformed.has_colors():
                    pcd_o3d.colors = pcd_transformed.colors
                vis.update_geometry(pcd_o3d)
                vis.poll_events()
                vis.update_renderer()

            # 将3D坐标轴投影到2D图像平面
            axis_length = 0.2  # 坐标轴长度（米）
            axes_3d = np.float32([
                [axis_length, 0, 0],  # X轴
                [0, axis_length, 0],  # Y轴
                [0, 0, axis_length]  # Z轴
            ])
            imgpts, _ = cv2.projectPoints(
                axes_3d,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )
            imgpts = imgpts.astype(int).reshape(-1, 2)

            # 将原点投影到图像平面
            origin_world = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            origin_img, _ = cv2.projectPoints(
                origin_world,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )
            origin_img = tuple(origin_img.reshape(-1, 2).astype(int)[0])

            # 检查原点是否在图像范围内
            img_height, img_width = color_frame.height, color_frame.width
            if not (0 <= origin_img[0] < img_width and 0 <= origin_img[1] < img_height):
                print("原点超出图像范围")

            # 检查轴的点是否在图像范围内
            for i, pt in enumerate(imgpts):
                if not (0 <= pt[0] < img_width and 0 <= pt[1] < img_height):
                    print(f"坐标轴{i}点{tuple(pt)}超出图像范围")

            # 在颜色图像上绘制坐标轴
            img_with_axes = np.asanyarray(color_frame.get_data()).copy()
            img_with_axes = draw_axes(img_with_axes, origin_img, imgpts)

            # 标记原点
            cv2.circle(img_with_axes, origin_img, 5, (0, 0, 255), -1)  # 红点

            # 绘制已点击的点（如果有）
            if last_clicked_position is not None and last_clicked_world is not None:
                x, y = last_clicked_position
                # 将世界坐标系的点投影回图像
                imgpt, _ = cv2.projectPoints(
                    np.array([last_clicked_world], dtype=np.float32),
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs
                )
                imgpt = tuple(imgpt.reshape(-1, 2).astype(int)[0])
                cv2.circle(img_with_axes, imgpt, 5, (255, 255, 0), -1)  # 青色点
                cv2.putText(
                    img_with_axes,
                    f"({last_clicked_world[0]:.2f}, {last_clicked_world[1]:.2f}, {last_clicked_world[2]:.2f}) m",
                    (imgpt[0] + 10, imgpt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

            # 显示图像
            cv2.imshow('RealSense - World Axes', img_with_axes)

            # 显示深度图像（可选）
            depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale  # 转换为米
            # 归一化深度图像到0-255进行显示
            depth_image_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_display = np.uint8(depth_image_display)
            cv2.imshow('Depth Image', depth_image_display)

            # 退出条件：按下 'q' 或 'ESC' 键
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                print("按下退出键。退出程序。")
                break

    except KeyboardInterrupt:
        print("程序被用户中断。")
    except Exception as e:
        print(f"发生错误: {e}")
        input("按任意键退出...")
    finally:
        # 停止管道并关闭所有窗口
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == "__main__":
    main()