import rosbag2_py
import csv
import os
import matplotlib.pyplot as plt
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
import tf_transformations

# Function to extract data from a rosbag and save it to a CSV file
def extract_data_from_bag(bag_path, output_csv):
    """
    Extracts data from a ROS 2 bag and saves it to a CSV file.

    Args:
        bag_path (str): Path to the rosbag directory or database.
        output_csv (str): Path to the output CSV file.
    """
    # Initialize rosbag reader for sequential reading
    reader = rosbag2_py.SequentialReader()
    
    # Configure storage options and specify SQLite3 as the storage backend
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    
    # Configure converter options to use CDR format for serialization/deserialization
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    
    # Open the rosbag with the defined options
    reader.open(storage_options, converter_options)

    # Open the output CSV file in write mode
    with open(output_csv, mode="w", newline="") as csvfile:
        # Define the column headers for the CSV file
        fieldnames = ["topic", "timestamp", "x", "y", "theta"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the column headers to the CSV file
        writer.writeheader()

        # Loop through each message in the rosbag
        while reader.has_next():
            # Read the next message, which includes the topic, data, and timestamp
            (topic, data, t) = reader.read_next()
            
            # Filter messages from specific topics of interest
            if topic in ["/ekf", "/odom", "/ground_truth"]:
                # Deserialize the message into an Odometry object
                msg = deserialize_message(data, Odometry)
                
                # Extract the x and y position and orientation (theta)
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                quat = msg.pose.pose.orientation  # Extract quaternion for orientation
                _, _, theta = tf_transformations.euler_from_quaternion(
                    [quat.x, quat.y, quat.z, quat.w]
                )
                
                # Write the extracted data (topic, timestamp, x, y, theta) to the CSV file
                writer.writerow({"topic": topic, "timestamp": t, "x": x, "y": y, "theta": theta})
    
    # Print a message to confirm that data extraction is complete
    print(f"Data extracted to {output_csv}")


# Function to plot extracted data from the CSV file
def plot_data_superimposed(csv_file):
    """
    Plots data extracted from CSV with superimposed plots for each state component.

    Args:
        csv_file (str): Path to the CSV file containing extracted data.
    """
    # Dictionary to organize data by topic (keys: "ekf", "odom", "ground_truth")
    data = {"ekf": [], "odom": [], "ground_truth": []}

    # Open the CSV file for reading
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        
        # Read each row and group data by topic
        for row in reader:
            topic = row["topic"].strip("/")  # Remove leading "/" from the topic name
            data[topic].append(
                {
                    "timestamp": float(row["timestamp"]),  # Extract timestamp
                    "x": float(row["x"]),  # Extract x position
                    "y": float(row["y"]),  # Extract y position
                    "theta": float(row["theta"]),  # Extract orientation
                }
            )

    # Create superimposed plots for x, y, and theta over time
    plt.figure(figsize=(12, 8))

    # Plot x positions over time
    plt.subplot(3, 1, 1)
    for topic, entries in data.items():
        if entries:
            timestamps = [entry["timestamp"] for entry in entries]
            x = [entry["x"] for entry in entries]
            plt.plot(timestamps, x, label=f"{topic} - x")
    plt.ylabel("x position")
    plt.legend()

    # Plot y positions over time
    plt.subplot(3, 1, 2)
    for topic, entries in data.items():
        if entries:
            timestamps = [entry["timestamp"] for entry in entries]
            y = [entry["y"] for entry in entries]
            plt.plot(timestamps, y, label=f"{topic} - y")
    plt.ylabel("y position")
    plt.legend()

    # Plot orientation (theta) over time
    plt.subplot(3, 1, 3)
    for topic, entries in data.items():
        if entries:
            timestamps = [entry["timestamp"] for entry in entries]
            theta = [entry["theta"] for entry in entries]
            plt.plot(timestamps, theta, label=f"{topic} - theta")
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation (theta)")
    plt.legend()

    plt.suptitle("State Components Over Time (Superimposed)")
    plt.tight_layout()
    plt.show()

    # Create a superimposed plot of trajectories (x, y) for all topics
    plt.figure(figsize=(8, 8))
    for topic, entries in data.items():
        if entries:
            x = [entry["x"] for entry in entries]
            y = [entry["y"] for entry in entries]
            plt.plot(x, y, label=f"{topic} trajectory")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Robot Trajectories (Superimposed)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Define the path to the rosbag database and output CSV file
    bag_path = "/root/sesar_lab/src/test_1/test_1_0.db3"  # Replace with actual rosbag path
    output_csv = "extracted_data.csv"  # Define CSV output filename

    # Step 1: Extract data from rosbag and save to CSV
    extract_data_from_bag(bag_path, output_csv)

    # Step 2: Plot the extracted data
    plot_data_superimposed(output_csv)
