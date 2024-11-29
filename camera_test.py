import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import os

def load_transformation_vectors(r_v_path, t_v_path):
    """
    Load rotation and translation vectors and convert them to a homogeneous transformation matrix.
    """
    try:
        rotation_vector = np.load(r_v_path)
        translation_vector = np.load(t_v_path)
        print("Original Rotation Vector (rotation_vector):")
        print(rotation_vector)
        print("Original Translation Vector (translation_vector):")
        print(translation_vector)

        # Ensure the shape is (3, 1)
        rotation_vector = rotation_vector.reshape(3, 1)
        translation_vector = translation_vector.reshape(3, 1)

        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rotation_vector)
        R = R.astype(np.float64)
        print("Rotation Matrix R:")
        print(R)

        # Construct homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation_vector.flatten()
        print("Transformation Matrix T:")
        print(T)

        return T

    except Exception as e:
        print(f"Error loading transformation vectors: {e}")
        return None

def initialize_realsense_pipeline():
    """
    Initialize the RealSense pipeline and configure depth and color streams.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline, config

def create_open3d_visualizer():
    """
    Initialize the Open3D visualizer window and point cloud object.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Real-Time Point Cloud', width=640, height=480)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    return vis, pcd

def main(r_v_path, t_v_path):
    """
    Main function for real-time point cloud capturing, processing, and visualization.
    """
    # Load transformation matrix
    T = load_transformation_vectors(r_v_path, t_v_path)
    if T is None:
        print("Failed to load transformation matrix. Exiting program.")
        return

    # Initialize RealSense pipeline
    pipeline, config = initialize_realsense_pipeline()

    # Initialize Open3D visualizer
    vis, pcd = create_open3d_visualizer()

    # Create pointcloud object
    pc = rs.pointcloud()

    try:
        print("Starting real-time point cloud capture and visualization. Press 'Ctrl+C' to exit.")
        loop_count = 0
        while True:
            loop_count += 1
            print(f"\n--- Loop Iteration: {loop_count} ---")
            # Capture frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                print("Failed to retrieve depth or color frame, skipping loop.")
                continue

            print("Successfully captured depth and color frames.")

            # Align depth frame to color frame
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not aligned_color_frame:
                print("Failed to align depth or color frame, skipping loop.")
                continue

            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            print("Successfully aligned depth frame to color frame.")
            print(f"Depth image shape: {aligned_depth_image.shape}")
            print(f"Depth image min: {aligned_depth_image.min()}, max: {aligned_depth_image.max()}")

            # Generate point cloud and map to color frame
            pc.map_to(aligned_color_frame)
            points = pc.calculate(aligned_depth_frame)  # Returns rs.points object

            print("Successfully generated raw point cloud.")

            # Get vertices
            vertices_structured = np.asanyarray(points.get_vertices())
            print(f"Structured vertices array dtype: {vertices_structured.dtype}")
            print(f"Number of vertices: {vertices_structured.shape[0]}")
            print(f"Available fields in vertices_structured: {vertices_structured.dtype.names}")

            # Correctly extract x, y, z
            if 'x' in vertices_structured.dtype.names and \
               'y' in vertices_structured.dtype.names and \
               'z' in vertices_structured.dtype.names:
                x = vertices_structured['x'].astype(np.float64)
                y = vertices_structured['y'].astype(np.float64)
                z = vertices_structured['z'].astype(np.float64)
                vertices = np.stack((x, y, z), axis=-1)
            else:
                print("Missing required fields 'x', 'y', 'z' in vertices_structured, skipping loop.")
                continue

            print(f"Converted vertices shape: {vertices.shape}")
            print(f"First 5 vertices:\n{vertices[:5]}")

            # Convert to homogeneous coordinates (N,4)
            ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
            vertices_hom = np.hstack((vertices, ones))  # Shape: (N,4)

            # Apply transformation matrix to get real-world coordinates
            transformed_vertices_hom = vertices_hom @ T.T  # Shape: (N,4)
            transformed_vertices = transformed_vertices_hom[:, :3]  # Take first 3 columns

            print(f"Transformed vertices shape: {transformed_vertices.shape}")

            # Filter out invalid points
            valid_mask = np.isfinite(transformed_vertices).all(axis=1)
            transformed_vertices = transformed_vertices[valid_mask]
            if transformed_vertices.size == 0:
                print("All transformed points are invalid, skipping loop.")
                continue

            # Optionally, filter points within a specific range
            bbox = (transformed_vertices > -10) & (transformed_vertices < 10)
            valid_mask = bbox.all(axis=1)
            transformed_vertices = transformed_vertices[valid_mask]
            if transformed_vertices.size == 0:
                print("All transformed points are outside the bounding box, skipping loop.")
                continue

            # Adjust coordinate axes
            axis_transform = np.array([
                [1, 0,  0],
                [0, 0, 1],
                [0, -1, 0]
            ], dtype=np.float64)  # Shape: (3,3)
            transformed_vertices = transformed_vertices @ axis_transform.T

            print("Completed coordinate axis transformation.")
            print(f"First 5 transformed vertices:\n{transformed_vertices[:5]}")

            # Assign points and colors to Open3D PointCloud
            pcd.points = o3d.utility.Vector3dVector(transformed_vertices)
            pcd.colors = o3d.utility.Vector3dVector(np.ones((transformed_vertices.shape[0], 3)))  # White color

            # Update geometry and render
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Sleep to control frame rate (~30 FPS)
            time.sleep(0.033)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close visualizer and stop pipeline
        vis.destroy_window()
        pipeline.stop()

# Test function
def test_create_transformed_point_cloud():
    """
    Test the create_transformed_point_cloud function.
    """
    # Rotation and translation vectors file paths
    r_v_file = "data/rvec.npy"  # Replace with your rotation vector file path
    t_v_file = "data/tvec.npy"  # Replace with your translation vector file path

    # Verify files exist
    if not os.path.exists(r_v_file):
        print(f"Rotation vector file not found: {r_v_file}")
        return
    if not os.path.exists(t_v_file):
        print(f"Translation vector file not found: {t_v_file}")
        return

    # Call main function
    main(r_v_file, t_v_file)

# Example usage
if __name__ == "__main__":
    test_create_transformed_point_cloud()