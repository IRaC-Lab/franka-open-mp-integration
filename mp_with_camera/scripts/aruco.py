#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import csv

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# ROS
rospy.init_node('sphere_publisher', anonymous=True)
pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
rate = rospy.Rate(100)

# Marker
human_wrist = Marker()
human_wrist.header.frame_id = "panda_link0"
human_wrist.ns = "sphere_namespace"
human_wrist.id = 0
human_wrist.type = Marker.SPHERE
human_wrist.action = Marker.ADD

human_wrist.scale.x = 0.05
human_wrist.scale.y = 0.05
human_wrist.scale.z = 0.05

human_wrist.color.r = 0.0
human_wrist.color.g = 1.0
human_wrist.color.b = 0.0
human_wrist.color.a = 1.0

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Aruco 
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

aruco_initialized = False
T_marker_to_camera_fixed = None

# RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[color_intr.fx, 0, color_intr.ppx],
                          [0, color_intr.fy, color_intr.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(color_intr.coeffs)

# Robot Transform
angle = 90
rad = np.deg2rad(angle)
R_rotation = np.array([[1, 0, 0],
                       [0, np.cos(rad), -np.sin(rad)],
                       [0, np.sin(rad),  np.cos(rad)]])

translation_vector = np.array([0, -0.25, -0.02])

T_robot = np.eye(4)
T_robot[:3, :3] = R_rotation
T_robot[:3, 3] = translation_vector

# Vars
idx11 = [0, 0]
idx12 = [0, 0]
prev_result = None
p_results = []

try:
    while not rospy.is_shutdown():

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Pose
        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for idx, lm in enumerate(results.pose_landmarks.landmark):
                x = int(lm.x * color_image.shape[1])
                y = int(lm.y * color_image.shape[0])

                if idx == 11:
                    idx11 = [x, y]
                if idx == 12:
                    idx12 = [x, y]

        # Depth
        if not (0 <= idx11[0] < 1280 and 0 <= idx11[1] < 720 and
                0 <= idx12[0] < 1280 and 0 <= idx12[1] < 720):
            continue

        d11 = depth_image[idx11[1], idx11[0]] * depth_scale
        d12 = depth_image[idx12[1], idx12[0]] * depth_scale

        p11 = rs.rs2_deproject_pixel_to_point(intr, idx11, d11)
        p12 = rs.rs2_deproject_pixel_to_point(intr, idx12, d12)

        center = np.array([(p11[0] + p12[0]) / 2,
                           (p11[1] + p12[1]) / 2,
                           (p11[2] + p12[2]) / 2,
                           1.0])

        # Aruco 저장
        if not aruco_initialized:
            corners, ids, _ = cv2.aruco.detectMarkers(
                color_image, aruco_dict, parameters=parameters)

            if ids is not None and 30 in ids.flatten():
                idx = np.where(ids.flatten() == 30)[0][0]
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, 0.05, camera_matrix, dist_coeffs)

                rvec = rvecs[idx][0]
                tvec = tvecs[idx][0]

                R, _ = cv2.Rodrigues(rvec)
                R_inv = R.T
                t_inv = -R_inv @ tvec

                T_marker_to_camera_fixed = np.eye(4)
                T_marker_to_camera_fixed[:3, :3] = R_inv
                T_marker_to_camera_fixed[:3, 3] = t_inv

                aruco_initialized = True
                print("Aruco initialized!")

        if not aruco_initialized:
            cv2.imshow("Color", color_image)
            cv2.waitKey(1)
            continue

        # Transform
        K = T_robot @ (T_marker_to_camera_fixed @ center)

        result = K[:3]

        if prev_result is not None and abs(result[1] - prev_result[1]) > 0.4:
            result = prev_result

        prev_result = result

        print("x:", result[0], "y:", result[1], "z:", result[2])

        human_wrist.header.stamp = rospy.Time.now()
        human_wrist.pose.position = Point(result[0], result[1], result[2])
        human_wrist.pose.orientation.w = 1.0
        pub.publish(human_wrist)

        cv2.imshow("Color", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
