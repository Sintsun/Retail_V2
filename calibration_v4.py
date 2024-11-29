import pyrealsense2 as rs
import numpy as np
import cv2
import os
import open3d as o3d

def initialize_realsense():
    """
    初始化RealSense管道，啟用深度和RGB流，並獲取相機內參。

    :return: 包含以下內容的元組：
             - pipeline: RealSense管道對象
             - depth_scale: 深度比例（米/單位）
             - rgb_intrinsic_matrix: RGB相機內參矩陣（3x3）
             - distortion_coeffs: 畸變係數（5x1）
             - pointcloud: RealSense點雲對象
    """
    # 初始化RealSense管道和配置
    pipeline = rs.pipeline()
    config = rs.config()

    # 啟用深度和RGB流，分辨率設定為1280x720，幀率30
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 開始流傳輸
    profile = pipeline.start(config)

    # 獲取深度比例
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例: {depth_scale} 米/單位")

    # 獲取RGB相機內參
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    # 創建RGB相機內參矩陣
    rgb_intrinsic_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    print("RGB 相機內參矩陣:\n", rgb_intrinsic_matrix)

    # 獲取畸變係數
    distortion_coeffs = np.array(intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)
    print("畸變係數:\n", distortion_coeffs)

    # 初始化點雲對象
    pointcloud = rs.pointcloud()

    return pipeline, depth_scale, rgb_intrinsic_matrix, distortion_coeffs, pointcloud

def generate_object_points(checkerboard_size, square_size):
    """
    生成棋盤格在世界坐標系中的3D物體點。

    :param checkerboard_size: 棋盤格內角點的列數和行數 (列, 行)
    :param square_size: 每個方塊的大小（米）
    :return: 3D點的數組 (列*行, 3)
    """
    objp = np.zeros((checkerboard_size[1] * checkerboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                           0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def capture_single_chessboard_image(pipeline, pointcloud, checkerboard_size=(5, 4)):
    """
    捕捉包含棋盤格的單張圖像並檢測角點。

    :param pipeline: RealSense管道對象
    :param pointcloud: RealSense點雲對象
    :param checkerboard_size: 棋盤格內角點的列數和行數 (列, 行)
    :return: 包含以下內容的元組：
             - corners_refined: 檢測並細化的棋盤格角點（numpy數組）
             - aligned_depth: 對齊的深度幀
             - color_image: 繪製角點的RGB圖像
    """
    try:
        while True:
            frames = pipeline.wait_for_frames()

            # 將深度幀對齊到RGB幀
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)

            aligned_depth = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth or not color_frame:
                continue

            # 映射點雲到RGB幀並計算點雲
            pointcloud.map_to(color_frame)
            points = pointcloud.calculate(aligned_depth)

            # 將RGB幀轉換為numpy數組
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 增強圖像對比度
            gray = cv2.equalizeHist(gray)

            # 查找棋盤格角點
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)

            if ret:
                print("成功檢測到棋盤格角點。")
                # 細化角點精度
                criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # 在圖像上繪製角點
                cv2.drawChessboardCorners(color_image, checkerboard_size, corners_refined, ret)

                # 顯示圖像
                cv2.imshow('棋盤格檢測', color_image)
                print("按 's' 鍵保存數據，或按 'q' 鍵退出。")

                key = cv2.waitKey(0)
                if key & 0xFF == ord('s'):
                    # 保存捕捉到的數據
                    save_dir = 'single_chessboard_image'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # 保存帶有角點的RGB圖像
                    cv2.imwrite(os.path.join(save_dir, 'chessboard_rgb.png'), color_image)
                    # 保存深度圖像為numpy數組
                    depth_image = np.asanyarray(aligned_depth.get_data())
                    np.save(os.path.join(save_dir, 'chessboard_depth.npy'), depth_image)
                    # 保存細化後的角點
                    np.save(os.path.join(save_dir, 'chessboard_corners.npy'), corners_refined)
                    print("已捕捉並保存棋盤格數據。")
                    return corners_refined, aligned_depth, color_image
                elif key & 0xFF == ord('q'):
                    print("退出程序。")
                    return None, None, None
            else:
                cv2.imshow('棋盤格檢測', color_image)
                print("未能檢測到棋盤格角點。")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("退出程序。")
                    return None, None, None

    finally:
        cv2.destroyAllWindows()

def compute_3d_points(corners, depth_frame, depth_scale, rgb_intrinsics):
    """
    使用深度數據將2D圖像點轉換為3D相機坐標。

    :param corners: 檢測到的2D圖像點 (Nx2)
    :param depth_frame: 對齊的深度幀
    :param depth_scale: 深度比例（米/單位）
    :param rgb_intrinsics: RGB相機內參矩陣 (3x3)
    :return: 3D點的數組 (Nx3)
    """
    points_3d = []
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for corner in corners:
        x, y = corner.ravel()
        x_int, y_int = int(round(x)), int(round(y))
        # 確保坐標在圖像邊界內
        if x_int < 0 or y_int < 0 or x_int >= width or y_int >= height:
            continue
        depth = depth_frame.get_distance(x_int, y_int) * depth_scale  # 轉換為米
        if depth == 0:
            continue  # 跳過無效深度
        # 計算實際世界坐標
        X = (x - rgb_intrinsics[0, 2]) * depth / rgb_intrinsics[0, 0]
        Y = (y - rgb_intrinsics[1, 2]) * depth / rgb_intrinsics[1, 1]
        Z = depth
        points_3d.append([X, Y, Z])

    return np.array(points_3d, dtype=np.float32)

def estimate_pose_with_icp(objp_world, objp_camera):
    """
    使用ICP算法估計姿態（旋轉和平移向量）。

    :param objp_world: 世界坐標系中的3D點 (Nx3)
    :param objp_camera: 相機坐標系中的3D點 (Nx3)
    :return: 包含旋轉向量 (rvec) 和平移向量 (tvec) 的元組
    """
    # 創建Open3D點雲對象
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(objp_world)
    target.points = o3d.utility.Vector3dVector(objp_camera)

    # 初始對齊（單位矩陣）
    threshold = 0.02  # 最大對齊距離2厘米
    trans_init = np.eye(4)

    # 執行ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # 提取變換矩陣
    transformation = reg_p2p.transformation
    rotation_matrix = transformation[:3, :3]
    translation_vector = transformation[:3, 3]

    # 將旋轉矩陣轉換為旋轉向量
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    tvec = translation_vector.reshape(3, 1)

    print("旋轉向量 (rvec):\n", rvec)
    print("平移向量 (tvec):\n", tvec)
    print("ICP匹配誤差:", reg_p2p.inlier_rmse)
    return rvec, tvec

def save_vectors(rvec, tvec, rgb_intrinsic_matrix, output_folder="./data"):
    """
    保存旋轉向量、平移向量和相機內參矩陣為.npy文件。

    :param rvec: 旋轉向量
    :param tvec: 平移向量
    :param rgb_intrinsic_matrix: RGB相機內參矩陣
    :param output_folder: 輸出文件夾路徑
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    np.save(os.path.join(output_folder, "rotation_vectors.npy"), rvec)
    np.save(os.path.join(output_folder, "translation_vectors.npy"), tvec)
    np.save(os.path.join(output_folder, "rgb_intrinsic_matrix.npy"), rgb_intrinsic_matrix)
    print(f"旋轉向量、平移向量和RGB內參矩陣已保存至 '{output_folder}' 文件夾。")

def draw_axes_on_image(img, rgb_intrinsic_matrix, dist_coeffs, rvec, tvec, axis_length=0.2):
    """
    在圖像上繪製3D坐標軸以可視化姿態。

    :param img: RGB圖像
    :param rgb_intrinsic_matrix: RGB相機內參矩陣
    :param dist_coeffs: 畸變係數
    :param rvec: 旋轉向量
    :param tvec: 平移向量
    :param axis_length: 坐標軸長度（米）
    :return: 繪製坐標軸後的圖像
    """
    # 定義3D坐標軸點
    axis = np.float32([
        [axis_length, 0, 0],  # X軸（紅色）
        [0, axis_length, 0],  # Y軸（綠色）
        [0, 0, axis_length]   # Z軸（藍色）
    ]).reshape(-1, 3)

    # 將3D點投影到2D圖像點
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    # 定義原點（假設第一個物體點為原點）
    origin_pt, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    origin_pt = tuple(origin_pt.reshape(-1, 2).astype(int)[0])

    # 繪製坐標軸
    img = cv2.line(img, origin_pt, tuple(imgpts[0]), (0, 0, 255), 3)  # X軸為紅色
    img = cv2.line(img, origin_pt, tuple(imgpts[1]), (0, 255, 0), 3)  # Y軸為綠色
    img = cv2.line(img, origin_pt, tuple(imgpts[2]), (255, 0, 0), 3)  # Z軸為藍色

    return img

def visualize_correspondence(objp_world, objp_camera, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs, img):
    """
    可視化3D物體點與其投影點之間的對應關係。

    :param objp_world: 世界坐標系中的3D點 (Nx3)
    :param objp_camera: 相機坐標系中的3D點 (Nx3)
    :param rvec: 旋轉向量
    :param tvec: 平移向量
    :param rgb_intrinsic_matrix: RGB相機內參矩陣
    :param dist_coeffs: 畸變係數
    :param img: 要繪製的圖像
    """
    # 將3D點投影回2D圖像點
    imgpoints2, _ = cv2.projectPoints(objp_world, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    imgpoints2 = imgpoints2.reshape(-1, 2).astype(int)

    # 繪製原始檢測點和投影點
    for i, (pt1, pt2) in enumerate(zip(objp_world, imgpoints2)):
        # 將3D點轉換為2D點（假設已知相機參數和姿態）
        pt_original = (int(pt2[0]), int(pt2[1]))
        pt_projected = tuple(pt2)
        cv2.circle(img, pt_original, 5, (0, 255, 0), -1)  # 原始點綠色
        cv2.circle(img, pt_projected, 3, (0, 0, 255), -1)  # 投影點紅色
        cv2.line(img, pt_original, pt_projected, (255, 0, 0), 1)  # 連線為藍色

    cv2.imshow('對應關係可視化', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    主函數，執行以下步驟：
    1. 初始化RealSense相機。
    2. 生成物體點。
    3. 捕捉單張棋盤格圖像並檢測角點。
    4. 使用深度數據計算3D點。
    5. 使用ICP估計姿態。
    6. 保存旋轉向量、平移向量和內參矩陣。
    7. 可視化對應關係。
    8. 在圖像上繪製3D坐標軸。
    """
    # 定義棋盤格參數
    checkerboard_size = (5, 4)  # 棋盤格內角點的列數和行數
    square_size = 0.13  # 每個方塊的大小（米）

    # 初始化RealSense管道和點雲對象，獲取內參和畸變係數
    pipeline, depth_scale, rgb_intrinsic_matrix, distortion_coeffs, pointcloud = initialize_realsense()

    # 生成世界坐標系中的3D點
    objp_world = generate_object_points(checkerboard_size, square_size)

    # 捕捉棋盤格圖像並檢測角點
    corners_refined, aligned_depth, color_image = capture_single_chessboard_image(pipeline, pointcloud, checkerboard_size)
    if corners_refined is None:
        print("未檢測到棋盤格角點。程序退出。")
        pipeline.stop()
        return

    # 使用深度數據計算3D點
    objp_camera = compute_3d_points(corners_refined, aligned_depth, depth_scale, rgb_intrinsic_matrix)
    if objp_camera.shape[0] < 4:
        print("有效3D點不足。程序退出。")
        pipeline.stop()
        return

    # 確保世界點和相機點的數量匹配
    objp_world_matched = objp_world[:objp_camera.shape[0]]

    # 使用ICP估計旋轉和平移向量
    rvec, tvec = estimate_pose_with_icp(objp_world_matched, objp_camera)

    # 保存旋轉向量、平移向量和RGB內參矩陣
    output_folder = "./data"
    save_vectors(rvec, tvec, rgb_intrinsic_matrix, output_folder)

    # 可視化3D點與其投影點之間的對應關係
    # 這裡假設objp_world已經是3D點，實際上可以根據需要進行調整
    visualize_correspondence(objp_world_matched, objp_camera, rvec, tvec, rgb_intrinsic_matrix, distortion_coeffs, color_image.copy())

    # 在圖像上繪製3D坐標軸以驗證姿態估計
    color_image_with_axes = draw_axes_on_image(color_image, rgb_intrinsic_matrix, distortion_coeffs, rvec, tvec,
                                               axis_length=0.2)  # 增加軸長度以提高可見性

    # 顯示帶有坐標軸的結果圖像
    cv2.imshow('姿態估計結果', color_image_with_axes)
    print("按任意鍵退出。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 停止RealSense管道
    pipeline.stop()

if __name__ == "__main__":
    main()