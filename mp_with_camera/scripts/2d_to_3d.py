#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
align_to = rs.stream.color

align = rs.align(align_to)

# 깊이 스케일 가져오기
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 카메라 인트린식 파라미터 가져오기
profile = pipeline.get_active_profile()
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()

        if frames == True:
            print("camera ON")
        else:
            print("camera OFF")

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 깊이 이미지 및 컬러 이미지 가져오기
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 관심있는 2D 이미지 좌표 (예: 중앙 픽셀)
        x = 320  # x 좌표
        y = 240  # y 좌표

        # 깊이 값 가져오기 (미터 단위로 변환)
        depth = depth_image[y, x] * depth_scale

        # 픽셀 좌표를 3D 월드 좌표로 변환
        depth_point = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth)

        # 결과 출력
        print(f"2D 좌표: ({x}, {y}), 깊이: {depth} m, 3D 좌표: X: {depth_point[0]} m, Y: {depth_point[1]} m, Z: {depth_point[2]} m")

        # 컬러 이미지 표시 (선택 사항)
        cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Color Image', color_image)

        if cv2.waitKey(1) == 27:
            break

finally:
    # 파이프라인 종료
    pipeline.stop()
    cv2.destroyAllWindows()
