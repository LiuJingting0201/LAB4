import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist  
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
import tf_transformations

from ament_index_python.packages import get_package_share_directory
import yaml
import os
import numpy as np
import math
import matplotlib.pyplot as plt

class EKF_localization_1(Node):
    def __init__(self):
        super().__init__('ekf_localization_1')

        # Initialize parameters for initial state and epsilon
        self.declare_parameter('initial_x_position', 0.0)
        self.declare_parameter('initial_y_position', 0.0)
        self.declare_parameter('initial_theta_position', 0.0)
        self.declare_parameter('zero_division_small_value', 10e-9)

        # Prediction rate
        self.declare_parameter('prediction_step_rate', 5)

        # Landmarks YAML file path
        self.declare_parameter('landmark_file', 'turtlebot3_perception/config/landmarks.yaml')
        

        # Initialize state variables
        self.initial_state_x = self.get_parameter('initial_x_position').value
        self.initial_state_y = self.get_parameter('initial_y_position').value
        self.initial_state_theta = self.get_parameter('initial_theta_position').value
        self.prediction_rate = self.get_parameter('prediction_step_rate').value
        self.epsilon = self.get_parameter('zero_division_small_value').value

        # Load landmarks
        landmark_file_path = self.get_parameter('landmark_file').value
        self.landmarks = self.load_landmarks(landmark_file_path)

        self.covariance = np.diag([0.1, 0.1, 0.1])  # Covariance Matrix Initialized
        self.linear_vel_x = 0.0  # Linear velocity
        self.angular_vel_w = 0.0  # Angular velocity

        self.time_stamp = self.get_clock().now().to_msg()


        # Subscriptions 
        self.cmdvel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        #if we want to take measurement from the teleops command
        self.landmark_sub = self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)
        # subscription to the landmark topic to get the id, range and bearing
        self.ground_truth = self.create_subscription(Odometry, '/ground_truth', self.get_ground_truth, 10)
        # To get actual odometry data from the robot

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)


        # Publisher for EKF estimated position
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        
        # Timer for periodic prediction step
        self.timer_period = 1 / self.prediction_rate  # seconds
        self.prediciton_time = self.create_timer(self.timer_period, self.timer_callback)



    def get_ground_truth(self,msg):
    # Extract ground truth position and orientation
        self.ground_x = msg.pose.pose.position.x
        self.ground_y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        _, _, self.ground_theta = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])



    def load_landmarks(self, landmark_file_path):
        # package_path = get_package_share_directory('turtlebot3_perception')
        # 
        file_path = "/root/sesar_lab/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml"

        # Load and parse the YAML file
        with open(file_path, 'r') as file:
            landmarks_data = yaml.safe_load(file)
        
        # Extract landmark positions
        landmarks = {}
        ids = landmarks_data['landmarks']['id']
        x_coords = landmarks_data['landmarks']['x']
        y_coords = landmarks_data['landmarks']['y']
        z_coords = landmarks_data['landmarks']['z']

        # Store landmarks in dictionary
        for i in range(len(ids)):
            landmark_id = ids[i]  
            x = x_coords[i]      
            y = y_coords[i]      
            z = z_coords[i]      

            landmarks[landmark_id] = {'x': x, 'y': y, 'z': z}

        return landmarks



    def cmd_vel_callback(self, msg):
        # Use linear and angular velocities from /cmd_vel for the EKF prediction (teleops)
        self.linear_vel_x = msg.linear.x
        self.angular_vel_w = msg.angular.z



    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        _, _, theta = tf_transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )
        self.track_odom.append({"x": x, "y": y, "theta": theta})



    def timer_callback(self):
        # Prediction step using the velocity motion model
        dt = self.timer_period  # Small time step
        x, y, theta = self.initial_state_x, self.initial_state_y, self.initial_state_theta
        v, w = self.linear_vel_x, self.angular_vel_w

        # Handle straight-line motion when angular velocity is near zero
        if abs(self.angular_vel_w) < self.epsilon:
            x_prime = x + v * dt * math.cos(theta)
            y_prime = y + v * dt * math.sin(theta)
            theta_prime = theta  # No change in orientation
        else:
            # Calculate new state with angular motion
            R = v / w  #Turning radius
            x_prime = x - R * math.sin(theta) + R * math.sin(theta + w * dt)
            y_prime = y + R * math.cos(theta) - R * math.cos(theta + w * dt)
            theta_prime = theta + w * dt

        # Compute Jacobians Gt and Vt
        Gt = np.array([
            [1, 0, -v * dt * math.sin(theta)],
            [0, 1, v * dt * math.cos(theta)],
            [0, 0, 1]
        ])

        Vt = np.array([
            [dt * math.cos(theta), 0],
            [dt * math.sin(theta), 0],
            [0, dt]
        ])

        # Process noise covariance
        Q = np.diag([0.1, 0.1])

        # Update covariance
        self.covariance = Gt @ self.covariance @ Gt.T + Vt @ Q @ Vt.T

        # Update state
        self.initial_state_x, self.initial_state_y, self.initial_state_theta = x_prime, y_prime, theta_prime


        odometry_msg = Odometry()

        odometry_msg.header.frame_id = "map"
        odometry_msg.header.stamp = self.time_stamp
        odometry_msg.pose.pose.position.x = self.initial_state_x
        odometry_msg.pose.pose.position.y = self.initial_state_y
        odometry_msg.pose.pose.orientation.z = math.sin(self.initial_state_theta / 2.0)
        odometry_msg.pose.pose.orientation.w = math.cos(self.initial_state_theta / 2.0)
        self.ekf_pub.publish(odometry_msg)
        self.get_logger().info(f'x:{odometry_msg.pose.pose.position.x},y:{odometry_msg.pose.pose.position.y}')



    def landmark_callback(self, msg):
        for landmark in msg.landmarks:
            # Extract measurements
            range_measured = landmark.range
            bearing_measured = landmark.bearing
            landmark_id = landmark.id
            
            landmark_pos = self.landmarks.get(landmark_id, None)
            if landmark_pos:
                landmark_x, landmark_y = landmark_pos['x'], landmark_pos['y']
                
                # Predicted range and bearing based on current state estimate
                dx = landmark_x - self.initial_state_x
                dy = landmark_y - self.initial_state_y
                predicted_range = math.sqrt(dx**2 + dy**2)
                predicted_bearing = math.atan2(dy, dx) - self.initial_state_theta
                
                # Measurement residuals (innovations)
                range_residual = range_measured - predicted_range
                bearing_residual = bearing_measured - predicted_bearing
                bearing_residual = (bearing_residual + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
                
                # Jacobian matrix for measurement model
                H = np.array([
                    [-dx / predicted_range, -dy / predicted_range, 0],
                    [dy / (predicted_range**2), -dx / (predicted_range**2), -1]
                ])

                # Measurement covariance matrix (example values, should be adjusted based on sensor characteristics)
                R = np.diag([0.1, 0.1])  # Assume range and bearing variances

                # Compute Kalman gain
                S = H @ self.covariance @ H.T + R  # Innovation covariance
                K = self.covariance @ H.T @ np.linalg.inv(S)  # Kalman gain

                # Update state estimate using measurement residuals
                innovation = np.array([range_residual, bearing_residual])
                state_update = K @ innovation
                self.initial_state_x += state_update[0]
                self.initial_state_y += state_update[1]
                self.initial_state_theta += state_update[2]

                # Update covariance matrix
                I = np.eye(3)
                self.covariance = (I - K @ H) @ self.covariance



def main(args=None):
    rclpy.init(args=args)
    ekf_localization_1 = EKF_localization_1()  # Initialize your EKF node
    try:
        rclpy.spin(ekf_localization_1)  # Keep the node running and listening for events
    except KeyboardInterrupt:
        pass  # Exit gracefully on Ctrl+C
    finally:
        rclpy.try_shutdown()  # Shut down the ROS client library

if __name__ == '__main__':
    main()