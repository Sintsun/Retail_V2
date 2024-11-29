import pyrealsense2 as rs

def get_laser_power_range():
    """
    Initializes the RealSense pipeline, retrieves the laser power range from the depth sensor,
    and prints the minimum, maximum, step size, and default laser power values.
    """
    # Create a RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure the pipeline to stream depth data
    # You can adjust the resolution and framerate as needed
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline
    try:
        print("Starting RealSense pipeline...")
        pipeline_profile = pipeline.start(config)
    except Exception as e:
        print(f"Failed to start RealSense pipeline: {e}")
        return

    try:
        # Access the depth sensor
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()

        # Check if the 'laser_power' option is supported
        if not depth_sensor.supports(rs.option.laser_power):
            print("The 'laser_power' option is not supported on this device.")
            return

        # Retrieve the laser power range
        laser_power_range = depth_sensor.get_option_range(rs.option.laser_power)

        # Display the laser power range details
        print("\n=== Laser Power Range Information ===")
        print(f"Minimum Laser Power: {laser_power_range.min}")
        print(f"Maximum Laser Power: {laser_power_range.max}")
        print(f"Laser Power Step Size: {laser_power_range.step}")
        print(f"Default Laser Power: {laser_power_range.def_val}")
        print("=====================================\n")

    except Exception as e:
        print(f"An error occurred while retrieving laser power range: {e}")

    finally:
        # Stop the pipeline to release resources
        print("Stopping RealSense pipeline...")
        pipeline.stop()

if __name__ == "__main__":
    get_laser_power_range()