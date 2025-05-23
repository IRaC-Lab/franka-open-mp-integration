#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import csv

import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# Initialize the ROS node
rospy.init_node('sphere_publisher', anonymous=True)

# Create a publisher for visualization markers
pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

# Set the rate for publishing
rate = rospy.Rate(100)  # Hz

p_results = []

# Create a Marker message
human_wrist = Marker()
human_wrist.header.frame_id = "panda_link0"  # Change as needed
human_wrist.header.stamp = rospy.Time.now()
human_wrist.ns = "sphere_namespace"
human_wrist.id = 0
human_wrist.type = Marker.SPHERE  # Specify the shape as a cube
human_wrist.action = Marker.ADD


# Set the scale of the sphere
human_wrist.scale.x = 0.05  # Diameter
human_wrist.scale.y = 0.05  # Diameter
human_wrist.scale.z = 0.05  # Diameter

# Set the color (RGBA)
human_wrist.color.r = 0.0  # 빨간색
human_wrist.color.g = 1.0  # 초록색
human_wrist.color.b = 0.0  # 파란색
human_wrist.color.a = 1.0  # 알파 (불투명도)

# Set MediaPipe Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Aruco 마커 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # 사전 가져오는 방법 변경
parameters = cv2.aruco.DetectorParameters()

# Set RealSense PipeLine
prev_result1 = None 

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)

# 카메라 내부 파라미터 설정 (RealSense 카메라 캘리브레이션 값 필요)
profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color) 
intrinsics = color_stream.as_video_stream_profile().get_intrinsics() 

camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs) #np.zeros((5, 1))  # 왜곡 계수

# X축을 기준으로 도 회전하는 회전 행렬
angle = 90
radians = np.deg2rad(angle)
R_rotation = np.array([[1, 0, 0],
                       [0, np.cos(radians), -np.sin(radians)],
                       [0, np.sin(radians), np.cos(radians)]])
print(R_rotation)
# 이동 벡터
#translation_vector = np.array([0, 0.02, -0.25])

translation_vector = np.array([0, -0.25, -0.02])

# For 2D to 3D

align_to = rs.stream.color
align = rs.align(align_to)

# Get Depth Scale
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Get Camera Intrinsics Parameter
profile = pipeline.get_active_profile()
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

idx11 = [0,0]
idx12 = [0, 0]

try:
    while True:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Get Depth Img and Color Img
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # MediaPipe Pose Estimation
        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks:

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # normalized 좌표를 실제 이미지 좌표로 변환
                x = landmark.x
                y = landmark.y

                int_x = int(landmark.x * color_image.shape[1])
                int_y = int(landmark.y * color_image.shape[0])

                if idx == 11:
                    idx11 = [int_x, int_y]

                if idx == 12:
                    idx12 = [int_x, int_y]

                if ((landmark.x < 1) & (landmark.y < 1)):
                    cv2.putText(color_image, str(idx), (int_x, int_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)      

        # Get Depth Value
        if ((idx11[0] >=0 ) & (idx11[0] <= 1280) & (idx11[1] >=0 ) & (idx11[1] <= 720) & (idx12[0] >=0 ) & (idx12[0] <= 1280) & (idx12[1] >=0 ) & (idx12[1] <= 720)):

            depth11 = depth_image[idx11[1], idx11[0]] * depth_scale
            depth_point1 = rs.rs2_deproject_pixel_to_point(intr, [idx11[0], idx11[1]], depth11)

            depth12 = depth_image[idx12[1], idx12[0]] * depth_scale
            depth_point2 = rs.rs2_deproject_pixel_to_point(intr, [idx12[0], idx12[1]], depth12)

            #print(f"Val 12 좌표: ({idx12[0]}, {idx12[1]}), 깊이: {depth12} m, 3D 좌표: X: {depth_point2[0]} m, Y: {depth_point2[1]} m, Z: {depth_point2[2]} m")

            #print(f"Val 11 좌표: ({idx11[0]}, {idx11[1]}), 깊이: {depth11} m, 3D 좌표: X: {depth_point1[0]} m, Y: {depth_point1[1]} m, Z: {depth_point1[2]} m")

            cv2.circle(color_image, (idx11[0], idx11[1]), 5, (0, 0, 255), -1)
            cv2.circle(color_image, (idx12[0], idx12[1]), 5, (0, 0, 255), -1)

            corners, ids, rejected = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=parameters)

            if ids == 30:
                    # 마커 포즈 추정
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
                    
                    rvec = rvecs[0][0]
                    tvec = tvecs[0][0]

                    # 카메라 좌표계에서 마커 좌표계로 변환 행렬
                    R, _ = cv2.Rodrigues(rvec)  # 회전 벡터를 회전 행렬로 변환
                    T_camera_to_marker = np.hstack((R, tvec.reshape(3, 1)))
                    T_camera_to_marker = np.vstack((T_camera_to_marker, [0, 0, 0, 1]))  # 4x4 행렬로 변환

                    # 역변환 (마커 좌표계에서 카메라 위치 : R_inv)
                    R_inv = R.T
                    t_inv = -R_inv @ tvec
                    T_marker_to_camera = np.hstack((R_inv, t_inv.reshape(3, 1)))
                    T_marker_to_camera = np.vstack((T_marker_to_camera, [0, 0, 0, 1]))  # 4x4 행렬로 변환
                    
                    # R_rotation은 로봇에서 마커 변환행렬
                    R_robot = R_rotation
                    T_robot = translation_vector

                    # 변환 행렬 T 구성 (마커로 부터 로봇 위치 변환 행렬)
                    T = np.array([[R_robot[0, 0], R_robot[0, 1], R_robot[0, 2], T_robot[0]],
                                [R_robot[1, 0], R_robot[1, 1], R_robot[1, 2], T_robot[1]],
                                [R_robot[2, 0], R_robot[2, 1], R_robot[2, 2], T_robot[2]],
                                [0, 0, 0, 1]])

                    if depth_point2 is not None:
                        #print(f"로봇 좌표에 대한 카메라 위치: {T_robot}")
                        final_point = [((round(depth_point1[0], 4) + round(depth_point2[0], 4))/2),
                                       ((round(depth_point1[1], 4) + round(depth_point2[1], 4))/2),
                                       ((round(depth_point1[2], 4) + round(depth_point2[2], 4))/2)]

                        point = np.array([round(final_point[0], 4), round(final_point[1], 4), round(final_point[2], 4), 1])
                        #point = np.array([-0.5, 0, 1.0, 1])
                        K = T @ (T_marker_to_camera @ point)

                        v_shoulders = np.array([((round(depth_point1[0], 4) - round(depth_point2[0], 4))),
                                                ((round(depth_point1[1], 4) - round(depth_point2[1], 4))),
                                                ((round(depth_point1[2], 4) - round(depth_point2[2], 4)))])

                        v_shoulders_xy = np.array([v_shoulders[0], v_shoulders[1], 0])

                        # Compute a perpendicular vector in the X-Y plane
                        v_perp_xy = np.array([-v_shoulders_xy[1], v_shoulders_xy[0], 0])

                        # Normalize the perpendicular vector
                        v_perp_xy_norm = v_perp_xy / np.linalg.norm(v_perp_xy)

                        # Scale the perpendicular vector to 5 cm (0.05 m)
                        v_perp_scaled = 0.03 * v_perp_xy_norm

                        result1 = K[0] #+ v_perp_scaled[0]
                        result2 = K[1] #+ v_perp_scaled[1]
                        result3 = K[2] #+ v_perp_scaled[2]

                        if prev_result1 is not None and(abs(result2 - prev_result2) >= 0.4):
                            result1 = prev_result1
                            result2 = prev_result2
                            result3 = prev_result3

                        prev_result1 = result1
                        prev_result2 = result2
                        prev_result3 = result3

                        print("x: ", result1)
                        print("y: ", result2)
                        print("z: ", result3)

                        p_results.append([depth_point1[0], depth_point1[1], depth_point1[2],
                                          depth_point2[0], depth_point2[1], depth_point2[2],
                                          final_point[0], final_point[1], final_point[2],
                                          result1, result2, result3])
                        with open('output.csv', mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(['d11x', 'd11y', 'd11z','d12x', 'd12y', 'd12z','cen_x', 'cen_y', 'cen_z','x', 'y', 'z'])  # 헤더 작성
                            writer.writerows(p_results)  # 결과 값 저장

                        human_wrist.pose.position = Point(result1, result2, result3)  # Change the coordinates as needed
                        human_wrist.pose.orientation.w = 1.0 
                        pub.publish(human_wrist)
                
                        
        cv2.imshow('Color Image', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()