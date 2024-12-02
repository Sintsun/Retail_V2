import pyrealsense2 as rs
import numpy as np
import cv2
import os
import mediapipe as mp

def draw_axes(img, origin, imgpts):
    """
    Draws 3D coordinate axes and origin marker on the image.

    :param img: The image to draw on.
    :param origin: The image coordinates of the origin (tuple of ints).
    :param imgpts: The projected image coordinates of the axis points (numpy array of shape (3, 2)).
    :return: The image with axes and origin drawn.
    """
    imgpts = imgpts.astype(int)

    # Draw origin marker (red filled circle)
    cv2.circle(img, origin, 5, (0, 0, 255), -1)  # Red filled circle with radius 5

    # Define axis colors: X - Red, Y - Green, Z - Blue
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    for i, pt in enumerate(imgpts):
        pt_tuple = tuple(pt)
        color = colors[i]
        img = cv2.line(img, origin, pt_tuple, color, 3)

    return img

def load_vectors(rvec_path, tvec_path):
    """
    Loads rotation and translation vectors from .npy files.

    :param rvec_path: Path to the rotation vector file.
    :param tvec_path: Path to the translation vector file.
    :return: Rotation vector (rvec) and translation vector (tvec) as Float32 numpy arrays with shape (3,1).
    """
    if not os.path.exists(rvec_path):
        raise FileNotFoundError(f"Rotation vector file not found: {rvec_path}")
    if not os.path.exists(tvec_path):
        raise FileNotFoundError(f"Translation vector file not found: {tvec_path}")

    rvec = np.load(rvec_path).astype(np.float32).squeeze()
    tvec = np.load(tvec_path).astype(np.float32).squeeze()

    # Reshape to (3,1)
    rvec = rvec.reshape(3, 1) if rvec.shape == (3,) else rvec
    tvec = tvec.reshape(3, 1) if tvec.shape == (3,) else tvec

    return rvec.astype(np.float32), tvec.astype(np.float32)

def load_distortion_coeffs(dist_coeffs_path):
    """
    Loads distortion coefficients from a .npy file. Assumes no distortion if file does not exist.

    :param dist_coeffs_path: Path to the distortion coefficients file.
    :return: Distortion coefficients as a Float32 numpy array.
    """
    if not os.path.exists(dist_coeffs_path):
        return np.zeros((5,), dtype=np.float32)  # Typically, there are 5 distortion coefficients
    dist_coeffs = np.load(dist_coeffs_path).astype(np.float32).flatten()
    return dist_coeffs

def initialize_realsense():
    """
    Initializes the RealSense pipeline, enables depth and RGB streams, and retrieves camera intrinsics.

    :return: A tuple containing:
             - pipeline: RealSense pipeline object
             - depth_scale: Depth scale (meters/unit)
             - intrinsics: RealSense camera intrinsics object
             - distortion_coeffs: Distortion coefficients (5)
             - pointcloud: RealSense point cloud object
    """
    # Initialize RealSense pipeline and configuration
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and RGB streams with resolution 1280x720 and 30 FPS
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Get RGB camera intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    # Get distortion coefficients
    distortion_coeffs = np.array(intrinsics.coeffs, dtype=np.float32).flatten()

    # Initialize point cloud object
    pointcloud = rs.pointcloud()

    return pipeline, depth_scale, intrinsics, distortion_coeffs, pointcloud

def main():
    # Set the data folder path, ensure it contains calibration data
    image_folder = "./data"  # Replace with your data folder path
    rvec_path = os.path.join(image_folder, "rotation_vectors.npy")
    tvec_path = os.path.join(image_folder, "translation_vectors.npy")
    camera_matrix_path = os.path.join(image_folder, "rgb_intrinsic_matrix.npy")
    dist_coeffs_path = os.path.join(image_folder, "distortion_coeffs.npy")  # Distortion coefficients path

    # Load camera intrinsic matrix
    if not os.path.exists(camera_matrix_path):
        raise FileNotFoundError(f"Camera intrinsic matrix file not found: {camera_matrix_path}")
    camera_matrix = np.load(camera_matrix_path).astype(np.float32)

    # Load distortion coefficients
    dist_coeffs = load_distortion_coeffs(dist_coeffs_path)

    # Load rotation and translation vectors
    try:
        rvec, tvec = load_vectors(rvec_path, tvec_path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        input("Press any key to exit...")
        return

    # Initialize RealSense pipeline and point cloud object, get intrinsics and distortion coefficients
    pipeline, depth_scale, intrinsics, distortion_coeffs, pointcloud = initialize_realsense()

    # Create an align object to align depth frames to color frames
    align = rs.align(rs.stream.color)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,  # Set the number of faces to detect as needed
        refine_landmarks=True,  # Whether to detect key points for eyes and lips
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()

            # Align depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned depth and color frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Map point cloud to color frame and calculate point cloud
            pointcloud.map_to(color_frame)
            points = pointcloud.calculate(depth_frame)

            # Convert color frame to RGB image
            color_image = np.asanyarray(color_frame.get_data())
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Use MediaPipe to detect face landmarks
            results = face_mesh.process(image_rgb)

            # Convert back to BGR for OpenCV display
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Draw coordinate axes on the image (persistent)
            # Define world coordinate system's 3D coordinate axes
            axis_length = 0.2  # 20 centimeters
            axes_3d = np.float32([
                [0, 0, 0],            # Origin
                [axis_length, 0, 0], # X-axis
                [0, axis_length, 0], # Y-axis
                [0, 0, axis_length]  # Z-axis
            ])

            # Project 3D coordinate axes points to 2D image plane
            imgpts, _ = cv2.projectPoints(
                axes_3d,     # 3D coordinate axes points in world coordinates
                rvec,        # Rotation vector (world to camera)
                tvec,        # Translation vector (world to camera)
                camera_matrix,  # Camera intrinsics
                dist_coeffs     # Distortion coefficients
            )
            imgpts = imgpts.reshape(-1, 2).astype(int)

            # Define origin image coordinates
            origin_img = tuple(imgpts[0])

            # Define axis end image coordinates
            x_axis = tuple(imgpts[1])
            y_axis = tuple(imgpts[2])
            z_axis = tuple(imgpts[3])

            # Draw axes on the image
            image_bgr = draw_axes(image_bgr, origin_img, np.array([x_axis, y_axis, z_axis]))

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    # Extract landmark coordinates
                    img_height, img_width, _ = image_bgr.shape
                    landmarks = face_landmarks.landmark
                    points_3d_camera = []
                    for landmark in landmarks:
                        x = int(landmark.x * img_width)
                        y = int(landmark.y * img_height)
                        if x < 0 or y < 0 or x >= img_width or y >= img_height:
                            points_3d_camera.append([np.nan, np.nan, np.nan])
                            continue
                        z = depth_frame.get_distance(x, y) * depth_scale  # Convert to meters
                        if z == 0:
                            points_3d_camera.append([np.nan, np.nan, np.nan])
                            continue
                        # Use RealSense's deprojection function to compute 3D camera coordinates
                        point_camera = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_frame.get_distance(x, y))
                        points_3d_camera.append(point_camera)

                    points_3d_camera = np.array(points_3d_camera, dtype=np.float32)  # (468, 3)

                    # Convert camera coordinates to world coordinates
                    # Transformation: Xw = R_inv * (Xc - tvec)
                    R, _ = cv2.Rodrigues(rvec)  # From rotation vector to rotation matrix
                    R_inv = R.T  # Inverse of rotation matrix
                    tvec_inv = -R_inv @ tvec    # Inverse translation vector

                    points_3d_world = (R_inv @ points_3d_camera.T + tvec_inv).T  # (468, 3)

                    # Visualize selected key points (e.g., nose)
                    selected_indices = [1]  # Example landmark index (1 is usually the nose tip in MediaPipe)
                    for idx in selected_indices:
                        point_world = points_3d_world[idx]
                        if np.any(np.isnan(point_world)):
                            continue
                        # Display world coordinates on the image
                        # Determine text position (e.g., near the landmark)
                        x_pixel = int(landmarks[idx].x * img_width)
                        y_pixel = int(landmarks[idx].y * img_height)
                        text = f"X: {point_world[0]:.2f}m Y: {point_world[1]:.2f}m Z: {point_world[2]:.2f}m"
                        # Put text near the landmark with larger font size and thickness
                        cv2.putText(image_bgr, text, (x_pixel + 10, y_pixel - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the image with axes and annotated coordinates
            cv2.imshow('RealSense - World Axes', image_bgr)

            # Display the depth image with normalization and color mapping for better visibility
            depth_normalized = cv2.normalize(
                np.asanyarray(depth_frame.get_data()) * depth_scale,
                None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            cv2.imshow('Depth Image', depth_colormap)

            # Exit condition: press 'q' or 'ESC' key
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
        input("Press any key to exit...")
    finally:
        # Stop RealSense pipeline and close all windows
        face_mesh.close()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()