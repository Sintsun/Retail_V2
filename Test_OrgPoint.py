import pyrealsense2 as rs
import numpy as np
import cv2
import os

def draw_axes(img, origin, imgpts):
    imgpts = imgpts.astype(int)
    for i, pt in enumerate(imgpts):
        pt_tuple = tuple(pt.ravel())
        # Define axis colors: X - Red, Y - Green, Z - Blue
        color = (0, 0, 255) if i == 0 else (0, 255, 0) if i == 1 else (255, 0, 0)
        img = cv2.line(img, origin, pt_tuple, color, 3)
    return img

def load_vectors(rvec_path, tvec_path):
    if not os.path.exists(rvec_path):
        raise FileNotFoundError(f"Rotation vector file not found: {rvec_path}")
    if not os.path.exists(tvec_path):
        raise FileNotFoundError(f"Translation vector file not found: {tvec_path}")

    rvec = np.load(rvec_path)
    tvec = np.load(tvec_path)

    # Squeeze dimensions
    rvec = rvec.squeeze()
    tvec = tvec.squeeze()

    # Ensure shape is (3, 1)
    if rvec.shape == (1, 3):
        rvec = rvec.T
    elif rvec.shape == (3,):
        rvec = rvec.reshape(3, 1)
    elif rvec.shape != (3, 1):
        raise ValueError(f"Rotation vector must be shape (3,1) or (1,3), but got {rvec.shape}")

    if tvec.shape == (1, 3):
        tvec = tvec.T
    elif tvec.shape == (3,):
        tvec = tvec.reshape(3, 1)
    elif tvec.shape != (3, 1):
        raise ValueError(f"Translation vector must be shape (3,1) or (1,3), but got {tvec.shape}")

    return rvec.astype(np.float32), tvec.astype(np.float32)

def main():
    image_folder = r"./data"  # Replace with your image folder path
    rvec_path = os.path.join(image_folder, "rotation_vectors.npy")
    tvec_path = os.path.join(image_folder, "translation_vectors.npy")
    camera_matrix_path = os.path.join(image_folder, "camera_matrix.npy")

    # Load camera intrinsic matrix
    if not os.path.exists(camera_matrix_path):
        print(f"Camera matrix file not found: {camera_matrix_path}")
        input("Press any key to exit...")
        return
    camera_matrix = np.load(camera_matrix_path).astype(np.float32)

    # Assume no distortion
    dist_coeffs = np.zeros((4,1), dtype=np.float32)

    # Load rotation and translation vectors
    try:
        rvec, tvec = load_vectors(rvec_path, tvec_path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        input("Press any key to exit...")
        return

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start the pipeline
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("Unable to start RealSense pipeline:", e)
        input("Press any key to exit...")
        return

    # Create alignment object to align depth frames to color frames
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Initialize point cloud (if needed)
    pc = rs.pointcloud()

    # Define axis length (in millimeters, e.g., 150 mm = 15 cm)
    axis_length = 150  # 150 mm
    axes_3d = np.float32([
        [axis_length, 0, 0],  # X-axis (Red)
        [0, axis_length, 0],  # Y-axis (Green)
        [0, 0, axis_length]   # Z-axis (Blue)
    ])

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Generate point cloud and map to color frame
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # Convert images to NumPy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Project axis points
            imgpts, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.astype(int).reshape(-1, 2)

            # Project origin
            origin_world = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            origin_img, _ = cv2.projectPoints(origin_world, rvec, tvec, camera_matrix, dist_coeffs)
            origin_img = tuple(origin_img[0].ravel().astype(int))

            # Get image dimensions
            img_height, img_width = color_image.shape[:2]

            # Draw axes
            img_with_axes = draw_axes(color_image.copy(), origin_img, imgpts)

            # Mark origin
            cv2.circle(img_with_axes, origin_img, 5, (0, 0, 255), -1)  # Red dot

            # Enhance depth image visibility
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Concatenate color image with axes and depth map side by side
            combined_image = np.hstack((img_with_axes, depth_colormap))

            # Display the image
            cv2.imshow('RealSense - Color with Axes and Depth Map', combined_image)

            # Exit condition: press 'q' or 'ESC' key
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("An error occurred:", e)
        input("Press any key to exit...")
    finally:
        # Stop the pipeline and close windows
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()