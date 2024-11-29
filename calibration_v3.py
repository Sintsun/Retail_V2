import pyrealsense2 as rs
import numpy as np
import cv2
import os

def initialize_realsense():
    """
    Initialize RealSense pipeline, enable depth and color streams, and retrieve camera intrinsics.

    :return: tuple containing:
             - pipeline: RealSense pipeline object
             - depth_scale: Depth scale (meters/unit)
             - rgb_intrinsic_matrix: RGB camera intrinsic matrix (3x3)
             - distortion_coeffs: Distortion coefficients (5x1)
             - pointcloud: RealSense pointcloud object
    """
    # Initialize RealSense pipeline and configuration
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams with higher resolution
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} meters/unit")

    # Get RGB camera intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    # Create RGB camera intrinsic matrix
    rgb_intrinsic_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    print("RGB Camera Intrinsic Matrix:\n", rgb_intrinsic_matrix)

    # Get distortion coefficients
    distortion_coeffs = np.array(intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)
    print("Distortion Coefficients:\n", distortion_coeffs)

    # Initialize pointcloud object
    pointcloud = rs.pointcloud()

    return pipeline, depth_scale, rgb_intrinsic_matrix, distortion_coeffs, pointcloud

def generate_object_points(checkerboard_size, square_size):
    """
    Generate 3D object points for the chessboard in world coordinates.

    :param checkerboard_size: Tuple indicating number of inner corners per chessboard row and column (columns, rows)
    :param square_size: Size of a square in meters
    :return: Array of 3D points (columns*rows, 3)
    """
    objp = np.zeros((checkerboard_size[1] * checkerboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                           0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def capture_single_chessboard_image(pipeline, pointcloud, checkerboard_size=(5, 4)):
    """
    Capture a single image containing a chessboard and detect corners.

    :param pipeline: RealSense pipeline object
    :param pointcloud: RealSense pointcloud object
    :param checkerboard_size: Tuple indicating number of inner corners per chessboard row and column (columns, rows)
    :return: tuple containing:
             - corners_refined: Detected and refined chessboard corners (numpy array)
             - aligned_depth: Aligned depth frame
             - color_image: Color image with drawn corners
    """
    try:
        while True:
            frames = pipeline.wait_for_frames()

            # Align depth frame to color frame
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)

            aligned_depth = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth or not color_frame:
                continue

            # Map pointcloud to color frame and calculate pointcloud
            pointcloud.map_to(color_frame)
            points = pointcloud.calculate(aligned_depth)
            # Optional: Get pointcloud vertices (not used in this function)
            # vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

            # Convert color frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Enhance image contrast
            gray = cv2.equalizeHist(gray)

            # Find chessboard corners
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)

            if ret:
                print("Chessboard corners detected successfully.")
                # Refine corner accuracy
                criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw corners on image
                cv2.drawChessboardCorners(color_image, checkerboard_size, corners_refined, ret)

                # Display image
                cv2.imshow('Chessboard', color_image)
                print("Press 's' to save the corners, or 'q' to quit.")

                key = cv2.waitKey(0)
                if key & 0xFF == ord('s'):
                    # Save captured data
                    save_dir = 'single_chessboard_image'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # Save RGB image with drawn corners
                    cv2.imwrite(os.path.join(save_dir, 'chessboard_rgb.png'), color_image)
                    # Save depth image as numpy array
                    depth_image = np.asanyarray(aligned_depth.get_data())
                    np.save(os.path.join(save_dir, 'chessboard_depth.npy'), depth_image)
                    # Save refined corners
                    np.save(os.path.join(save_dir, 'chessboard_corners.npy'), corners_refined)
                    print("Captured and saved chessboard data.")
                    return corners_refined, aligned_depth, color_image
                elif key & 0xFF == ord('q'):
                    print("Exiting program.")
                    return None, None, None
            else:
                cv2.imshow('Chessboard', color_image)
                print("Failed to detect chessboard corners.")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting program.")
                    return None, None, None

    finally:
        cv2.destroyAllWindows()

def load_chessboard_data(save_dir='single_chessboard_image'):
    """
    Load captured chessboard data from saved files.

    :param save_dir: Directory where chessboard data is saved
    :return: tuple containing:
             - color_image: Color image with drawn corners
             - depth_image: Depth image as numpy array
             - corners_refined: Refined chessboard corners (numpy array)
    """
    color_image = cv2.imread(os.path.join(save_dir, 'chessboard_rgb.png'))
    depth_image = np.load(os.path.join(save_dir, 'chessboard_depth.npy'))
    corners_refined = np.load(os.path.join(save_dir, 'chessboard_corners.npy'))
    return color_image, depth_image, corners_refined

def estimate_pose(objp_world, imgp, rgb_intrinsic_matrix, dist_coeffs=np.zeros((5, 1))):
    """
    Estimate rotation and translation vectors using solvePnP.

    :param objp_world: 3D points in world coordinates (Nx3)
    :param imgp: 2D image points (Nx2)
    :param rgb_intrinsic_matrix: RGB camera intrinsic matrix (3x3)
    :param dist_coeffs: Distortion coefficients (5x1)
    :return: tuple containing rotation vector (rvec) and translation vector (tvec)
    """
    if len(objp_world) < 4 or len(imgp) < 4:
        raise ValueError("At least 4 valid 3D points and 2D points are required for pose estimation.")

    # Use SOLVEPNP_ITERATIVE method
    success, rvec, tvec = cv2.solvePnP(objp_world, imgp, rgb_intrinsic_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        raise ValueError("cv2.solvePnP failed to estimate pose.")

    print("Rotation Vector (rvec):\n", rvec)
    print("Translation Vector (tvec):\n", tvec)
    return rvec, tvec

def save_vectors(rvec, tvec, rgb_intrinsic_matrix, output_folder="./data"):
    """
    Save rotation vector, translation vector, and camera matrix to .npy files.

    :param rvec: Rotation vector
    :param tvec: Translation vector
    :param rgb_intrinsic_matrix: RGB camera intrinsic matrix
    :param output_folder: Output folder path
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    np.save(os.path.join(output_folder, "rotation_vectors.npy"), rvec)
    np.save(os.path.join(output_folder, "translation_vectors.npy"), tvec)
    np.save(os.path.join(output_folder, "rgb_intrinsic_matrix.npy"), rgb_intrinsic_matrix)
    print(f"Rotation vectors, Translation vectors, and RGB Intrinsic matrix saved to '{output_folder}' folder.")

def draw_axes_on_image(img, rgb_intrinsic_matrix, dist_coeffs, rvec, tvec, axis_length=0.2):
    """
    Draw 3D coordinate axes on the image to visualize pose.

    :param img: Color image
    :param rgb_intrinsic_matrix: RGB camera intrinsic matrix
    :param dist_coeffs: Distortion coefficients
    :param rvec: Rotation vector
    :param tvec: Translation vector
    :param axis_length: Length of the axes in meters
    :return: Image with drawn axes
    """
    # Define the 3D coordinates of the axes
    axis = np.float32([
        [axis_length, 0, 0],  # X axis (red)
        [0, axis_length, 0],  # Y axis (green)
        [0, 0, axis_length]   # Z axis (blue)
    ]).reshape(-1, 3)

    # Project the 3D points to 2D image points
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    # Define origin in image (assuming first object point is at origin)
    origin_pt, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    origin_pt = tuple(origin_pt.reshape(-1, 2).astype(int)[0])

    # Draw the axes
    img = cv2.line(img, origin_pt, tuple(imgpts[0]), (0, 0, 255), 3)  # X axis in red
    img = cv2.line(img, origin_pt, tuple(imgpts[1]), (0, 255, 0), 3)  # Y axis in green
    img = cv2.line(img, origin_pt, tuple(imgpts[2]), (255, 0, 0), 3)  # Z axis in blue

    return img

def compute_reprojection_error(objp_world, imgp, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs):
    """
    Compute the reprojection error to evaluate calibration accuracy.

    :param objp_world: 3D object points in world coordinates (Nx3)
    :param imgp: 2D image points (Nx2)
    :param rvec: Rotation vector
    :param tvec: Translation vector
    :param rgb_intrinsic_matrix: RGB camera intrinsic matrix (3x3)
    :param dist_coeffs: Distortion coefficients (5x1)
    :return: Mean reprojection error (float)
    """
    # Project the 3D points back to 2D image points
    imgpoints2, _ = cv2.projectPoints(objp_world, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)

    # Reshape imgpoints2 from (N, 1, 2) to (N, 2)
    imgpoints2 = imgpoints2.reshape(-1, 2).astype(np.float32)

    # Ensure imgp is also of type float32 and shape (N, 2)
    imgp = imgp.astype(np.float32)

    # Compute the L2 norm (Euclidean distance) between the two sets of points
    error = cv2.norm(imgp, imgpoints2, cv2.NORM_L2) / len(imgpoints2)

    print(f"Mean Reprojection Error: {error}")
    return error

def visualize_correspondence(objp_world, imgp, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs, img):
    """
    Visualize correspondence between 3D object points and their projections.

    :param objp_world: 3D object points in world coordinates (Nx3)
    :param imgp: 2D image points (Nx2)
    :param rvec: Rotation vector
    :param tvec: Translation vector
    :param rgb_intrinsic_matrix: RGB camera intrinsic matrix
    :param dist_coeffs: Distortion coefficients
    :param img: Image to draw on
    """
    imgpoints2, _ = cv2.projectPoints(objp_world, rvec, tvec, rgb_intrinsic_matrix, dist_coeffs)
    imgpoints2 = imgpoints2.reshape(-1, 2).astype(int)

    for i, (pt1, pt2) in enumerate(zip(imgp, imgpoints2)):
        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2)
        cv2.circle(img, pt1, 5, (0, 255, 0), -1)  # Original detected point in green
        cv2.circle(img, pt2, 3, (0, 0, 255), -1)  # Projected point in red
        cv2.line(img, pt1, pt2, (255, 0, 0), 1)    # Line connecting them
    cv2.imshow('Correspondence', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Main function to perform the following steps:
    1. Initialize RealSense camera.
    2. Generate object points.
    3. Capture a single chessboard image and detect corners.
    4. Estimate pose using solvePnP.
    5. Save rotation vectors, translation vectors, and intrinsic matrix.
    6. Compute reprojection error.
    7. Visualize correspondence.
    8. Draw 3D axes on the image.
    """
    # Define chessboard parameters
    checkerboard_size = (5, 4)  # Number of inner corners per chessboard column and row
    square_size = 0.13  # Size of a square in meters

    # Initialize RealSense pipeline and pointcloud object, get intrinsics and distortion coefficients
    pipeline, depth_scale, rgb_intrinsic_matrix, distortion_coeffs, pointcloud = initialize_realsense()

    # Generate 3D points in world coordinates
    objp_world = generate_object_points(checkerboard_size, square_size)

    # Capture a chessboard image and detect corners
    corners_refined, aligned_depth, color_image = capture_single_chessboard_image(pipeline, pointcloud, checkerboard_size)
    if corners_refined is None:
        print("Chessboard corners not detected. Exiting program.")
        pipeline.stop()
        return

    # No need to reload data since corners are already captured and refined
    # Proceed to use corners_refined directly as imgp
    imgp = corners_refined.reshape(-1, 2).astype(np.float32)

    # Estimate rotation and translation vectors using solvePnP
    try:
        rvec, tvec = estimate_pose(objp_world, imgp, rgb_intrinsic_matrix, distortion_coeffs)
    except ValueError as e:
        print(e)
        pipeline.stop()
        return

    # Save rotation vector, translation vector, and RGB intrinsic matrix
    output_folder = "./data"
    save_vectors(rvec, tvec, rgb_intrinsic_matrix, output_folder)

    # Compute and print reprojection error
    compute_reprojection_error(objp_world, imgp, rvec, tvec, rgb_intrinsic_matrix, distortion_coeffs)

    # Visualize correspondence between detected and projected points
    visualize_correspondence(objp_world, imgp, rvec, tvec, rgb_intrinsic_matrix, distortion_coeffs, color_image.copy())

    # Draw 3D coordinate axes on the image to verify pose estimation
    color_image_with_axes = draw_axes_on_image(color_image, rgb_intrinsic_matrix, distortion_coeffs, rvec, tvec,
                                               axis_length=0.2)  # Increased axis length for better visibility

    # Display the result image with axes
    cv2.imshow('Pose Estimation', color_image_with_axes)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Stop RealSense pipeline
    pipeline.stop()

if __name__ == "__main__":
    main()