import pyrealsense2 as rs
import numpy as np
import cv2
import os

def draw_axes(img, origin, imgpts):
    """
    在图像上绘制3D坐标轴。

    :param img: 要绘制的图像。
    :param origin: 原点的图像坐标。
    :param imgpts: 投影后的轴点的图像坐标。
    :return: 绘制坐标轴后的图像。
    """
    imgpts = imgpts.astype(int)
    print(f"原点（图像坐标）: {origin}, 类型: {type(origin)}")
    for i, pt in enumerate(imgpts):
        pt_tuple = tuple(pt.ravel())
        print(f"坐标轴{i}点（图像坐标）: {pt_tuple}, 类型: {type(pt_tuple)}")
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

    print(f"加载的rvec: {rvec}, 形状: {rvec.shape}")
    print(f"加载的tvec: {tvec}, 形状: {tvec.shape}")

    # 如果需要，调整为(3,1)形状
    rvec = rvec.reshape(3, 1) if rvec.shape == (3,) else rvec
    tvec = tvec.reshape(3, 1) if tvec.shape == (3,) else tvec

    print(f"最终的rvec形状: {rvec.shape}")
    print(f"最终的tvec形状: {tvec.shape}")
    print(f"旋转向量 (rvec):\n{rvec}")
    print(f"平移向量 (tvec):\n{tvec}")

    return rvec.astype(np.float32), tvec.astype(np.float32)

def load_distortion_coeffs(dist_coeffs_path):
    """
    从.npy文件中加载畸变系数。如果文件不存在，则假设没有畸变。

    :param dist_coeffs_path: 畸变系数文件路径。
    :return: 畸变系数的Float32类型numpy数组。
    """
    if not os.path.exists(dist_coeffs_path):
        print(f"未找到畸变系数文件: {dist_coeffs_path}. 假设没有畸变。")
        return np.zeros((5, 1), dtype=np.float32)  # 通常，畸变有5个系数
    dist_coeffs = np.load(dist_coeffs_path).astype(np.float32)
    print("畸变系数已加载。")
    return dist_coeffs

def main():
    image_folder = "./data"  # 请替换为您的数据文件夹路径
    rvec_path = os.path.join(image_folder, "rotation_vectors.npy")
    tvec_path = os.path.join(image_folder, "translation_vectors.npy")
    camera_matrix_path = os.path.join(image_folder, "rgb_intrinsic_matrix.npy")
    dist_coeffs_path = os.path.join(image_folder, "distortion_coeffs.npy")  # 畸变系数路径

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
        return

    # 初始化RealSense管道
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

    # 获取深度传感器的深度缩放单位（米/单位）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度缩放单位: {depth_scale} 米/单位")

    # **设置激光器功率为最大值（根据设备支持的最大值）**
    try:
        # 获取激光功率范围
        laser_power_range = depth_sensor.get_option_range(rs.option.laser_power)
        min_laser_power = laser_power_range.min
        max_laser_power = laser_power_range.max
        step_laser_power = laser_power_range.step

        print("\n=== 激光功率范围信息 ===")
        print(f"最小激光功率: {min_laser_power}")
        print(f"最大激光功率: {max_laser_power}")
        print(f"激光功率步长: {step_laser_power}")
        print("==========================\n")

        # 设定期望的激光功率
        desired_laser_power = max_laser_power  # 设置为最大值

        # 将期望激光功率四舍五入到最近的步长
        desired_laser_power = round(desired_laser_power / step_laser_power) * step_laser_power

        # 设置激光功率
        depth_sensor.set_option(rs.option.laser_power, desired_laser_power)
        print(f"激光功率已设置为: {desired_laser_power}")
    except Exception as e:
        print(f"设置激光功率失败: {e}")
        print("继续使用默认激光功率。")

    # 定义轴的长度（米）
    axis_length = 0.2  # 0.2米

    # 定义在世界坐标系中的3D坐标轴
    axes_3d = np.float32([
        [axis_length, 0, 0],  # X轴
        [0, axis_length, 0],  # Y轴
        [0, 0, axis_length]   # Z轴
    ])

    # 初始化点云对象
    pc = rs.pointcloud()
    points = rs.points()

    # 初始化用于点击和采样的变量
    vertices_global = None
    clicked_flag = False
    clicked_x, clicked_y = -1, -1
    last_clicked_position = None
    last_clicked_world = None
    SAMPLE_COUNT = 10
    sampling = False
    sample_counter = 0
    sampled_points = []

    # 定义鼠标点击回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_flag, clicked_x, clicked_y, sampling, sample_counter, sampled_points
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_flag = True
            clicked_x, clicked_y = x, y
            sampling = True
            sample_counter = 0
            sampled_points = []
            print(f"鼠标点击位置: ({x}, {y})")

    # 创建窗口并设置鼠标回调
    cv2.namedWindow('RealSense - World Axes')
    cv2.setMouseCallback('RealSense - World Axes', mouse_callback)

    try:
        while True:
            # 等待获取帧
            frames = pipeline.wait_for_frames()

            # 获取深度帧和颜色帧
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 将点云映射到颜色帧
            pc.map_to(color_frame)

            # 计算点云
            points = pc.calculate(depth_frame)

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

            # 获取图像尺寸
            img_height, img_width = color_frame.height, color_frame.width

            # 将3D坐标轴投影到2D图像平面
            imgpts, _ = cv2.projectPoints(
                axes_3d,  # 要投影的3D坐标轴
                rvec,  # 旋转向量
                tvec,  # 平移向量
                camera_matrix,  # 相机内参矩阵
                dist_coeffs  # 畸变系数
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

            # 更新全局顶点
            vertices_global = vertices

            # 处理深度图像进行显示
            depth_image = np.asanyarray(depth_frame.get_data()) * depth_scale  # 转换为米
            # 归一化深度图像到0-255进行显示
            depth_image_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_display = np.uint8(depth_image_display)

            # 如果处于采样模式，收集样本
            if sampling and vertices_global is not None:
                if sample_counter < SAMPLE_COUNT:
                    x, y = clicked_x, clicked_y
                    print(f"采样 {sample_counter + 1}/{SAMPLE_COUNT} 点: (u={x}, v={y})")
                    if 0 <= x < img_width and 0 <= y < img_height:
                        point_camera = vertices_global[y, x]
                        if np.all(np.isfinite(point_camera)) and not np.all(point_camera == 0):
                            # 将当前的点添加到样本中
                            sampled_points.append(point_camera.reshape(3, 1))
                        else:
                            print("采样的相机坐标不合法。")
                    else:
                        print("采样点超出图像范围。")
                    sample_counter += 1
                else:
                    # 采样完成，计算平均值
                    if sampled_points:
                        avg_point_camera = np.mean(sampled_points, axis=0)  # 形状为 (3,1)

                        # 将旋转向量转换为旋转矩阵
                        R, _ = cv2.Rodrigues(rvec)  # R 是从世界到相机的旋转矩阵
                        R_inv = R.T  # R_inv 是从相机到世界的旋转矩阵
                        t_wc = -R_inv @ tvec  # 平移向量从相机到世界

                        # 将相机坐标系下的点转换为世界坐标系
                        point_world = R_inv @ avg_point_camera + t_wc

                        print(f"点击的像素点: (u={clicked_x}, v={clicked_y})")
                        print(
                            f"相机坐标系下的3D点（平均）: X={avg_point_camera[0][0]:.4f} m, Y={avg_point_camera[1][0]:.4f} m, Z={avg_point_camera[2][0]:.4f} m")
                        print(
                            f"世界坐标系下的3D点（平均）: X={point_world[0][0]:.4f} m, Y={point_world[1][0]:.4f} m, Z={point_world[2][0]:.4f} m")

                        # 存储最近点击的位置和世界坐标
                        last_clicked_position = (clicked_x, clicked_y)
                        last_clicked_world = point_world.copy()
                    else:
                        print("未收集到有效的采样点。")

                    # 重置采样变量
                    sampling = False
                    clicked_flag = False

            # 如果有点击点，重新绘制
            if 'last_clicked_position' in locals() and last_clicked_position is not None and \
               'last_clicked_world' in locals() and last_clicked_world is not None:
                x, y = last_clicked_position
                print(f"重新绘制最近点击的点: (u={x}, v={y})")
                cv2.circle(img_with_axes, (x, y), 5, (255, 255, 0), -1)  # 青色点
                cv2.putText(
                    img_with_axes,
                    f"X:{last_clicked_world[0][0]:.2f} Y:{last_clicked_world[1][0]:.2f} Z:{last_clicked_world[2][0]:.2f} m",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

            # 显示图像
            cv2.imshow('RealSense - World Axes', img_with_axes)
            cv2.imshow('Depth Image', depth_image_display)

            # 退出条件：按下'q'或'ESC'键
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

if __name__ == "__main__":
    main()