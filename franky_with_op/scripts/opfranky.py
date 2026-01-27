#!/usr/bin/env python3

import rospy, actionlib, sys
import numpy as np
from franky import Affine, CartesianMotion, Robot, ReferenceType, Gripper
from scipy.spatial.transform import Rotation
from std_msgs.msg import Float64MultiArray, Float64, Header

class franky_op:

    def __init__(self):
        rospy.init_node("franky_op")
        rate = rospy.Rate(100)

        try:
            self.robot = Robot("172.16.0.2")  
        except:
            self.robot = Robot("172.16.0.2", realtime=False)  
            rospy.logwarn("Running in non-realtime mode")

        self.robot.relative_dynamics_factor = 0.1
        self.state = self.robot.state

        self.gripper = Gripper("172.16.0.2")
        self.speed = 0.05
        self.force = 20.0

        # ee_pose 초기화
        cartesian_state = self.robot.current_cartesian_state
        robot_pose = cartesian_state.pose
        self.ee_pose = robot_pose.end_effector_pose

        # 응답지연 측정용 변수
        self.prev_position = None
        self.command_time = None
        self.command_seq = 0
        self.vel_threshold = 0.001  # 로그 출력 문제를 해결하기 위해 낮춤 (조정 가능)
        self.response_detected = False
        
        # dt 동적 계산용 이전 시간 초기화
        self.prev_time = rospy.Time.now().to_sec()

        # sub
        sub_pose = rospy.Subscriber('/op_mani/pose_list', Float64MultiArray, self.cb, queue_size=3)
        sub_gripper = rospy.Subscriber('/op_mani/pose_gripper', Float64, self.cb2, queue_size=3)
        sub_cmd_time = rospy.Subscriber('/op_mani/command_timestamp', Header, self.cmd_time_cb, queue_size=3)
        
        # pub
        self.pub = rospy.Publisher('/panda_csv', Float64MultiArray, queue_size=10)
        self.delay_pub = rospy.Publisher('/response_delay', Float64, queue_size=10)

    def cmd_time_cb(self, data):
        self.command_time = data.stamp.to_sec()
        self.command_seq = data.seq
        self.response_detected = False  

    def cb(self, data):
        # ee_pose 먼저 
        cartesian_state = self.robot.current_cartesian_state
        robot_pose = cartesian_state.pose
        self.ee_pose = robot_pose.end_effector_pose
        
        if len(data.data) == 7:
            quat = np.array([data.data[3],data.data[4],data.data[5],data.data[6]])
            motion = CartesianMotion(Affine([data.data[0], data.data[1], data.data[2]], quat), ReferenceType.Absolute)

            try:
                self.robot.move(motion, asynchronous=True)
                # 응답지연 측정
                self.measure_response_delay()
            except Exception as e:
                rospy.logerr(f"Robot move failed: {e}")
        else:
            print("wrong data")

        # CSV 퍼블리시
        panda_pose = Float64MultiArray()
        panda_pose.data = [self.ee_pose.translation[0], self.ee_pose.translation[1], self.ee_pose.translation[2]]
        self.pub.publish(panda_pose)
        rospy.sleep(0.001)

    def measure_response_delay(self):
        if self.response_detected or (rospy.Time.now().to_sec() - self.command_time < 0.1):  # 타이머 추가: 0.5초 내 중복 무시
            return
        
        current_position = np.array([self.ee_pose.translation[0], 
                                     self.ee_pose.translation[1], 
                                     self.ee_pose.translation[2]])
        
        # dt 동적 계산
        current_time = rospy.Time.now().to_sec()
        dt = max(current_time - self.prev_time, 0.005) if self.prev_time else 0.01
        
        if self.prev_position is not None and self.command_time is not None:
            velocity = np.linalg.norm(current_position - self.prev_position) / dt if dt > 0 else 0
            pos_change = np.linalg.norm(current_position - self.prev_position)
            
            # 디버그 로그 추가 (velocity와 dt 값 확인용)
            rospy.logdebug(f"Velocity: {velocity:.4f} m/s, dt: {dt:.4f} sec, Position Change: {pos_change:.4f} m")
            
            if velocity > self.vel_threshold and pos_change > 0.01:
                response_time = current_time
                delay = response_time - self.command_time
                
                rospy.loginfo(f"First Response Delay: {delay * 1000:.2f} ms")
                self.delay_pub.publish(delay)
                
                self.response_detected = True  # 플래그 설정
        
        # 이전 값 업데이트
        self.prev_position = current_position
        self.prev_time = current_time

    def cb2(self, data):
        gripper_width = min(max(data.data, 0.01), 0.105)
        success_future = self.gripper.move_async(gripper_width, self.speed)
        rospy.sleep(0.001)

if __name__ == '__main__':
    try:
        manipulator_t = franky_op()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

