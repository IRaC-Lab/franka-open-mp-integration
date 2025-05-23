#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 포즈 모델 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
pipeline.start(config)

try:
    while True:
        # 프레임 수집
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # RGB 데이터를 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # MediaPipe 포즈 추정
        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # 포즈 랜드마크 그리기
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 결과 출력

        if results.pose_landmarks:
            # 랜드마크 좌표 추출
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # normalized 좌표를 실제 이미지 좌표로 변환
                x = landmark.x
                y = landmark.y

                int_x = int(landmark.x * color_image.shape[1])
                int_y = int(landmark.y * color_image.shape[0])

                print(idx, x, y)

                if ((landmark.x < 1) & (landmark.y < 1)):
                    cv2.putText(color_image, str(idx), (int_x, int_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('RealSense + MediaPipe Pose Estimation', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 리소스 해제
    pipeline.stop()
    cv2.destroyAllWindows()
