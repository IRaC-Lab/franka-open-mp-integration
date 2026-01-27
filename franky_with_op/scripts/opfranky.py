#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, csv, os, numpy as np
from franky import Affine, CartesianMotion, Robot, ReferenceType, Gripper
from std_msgs.msg import Float64MultiArray, Float64, Bool, Header

class FrankyOP:
    def __init__(self):
        rospy.init_node('franky_op')

        # ───── 로봇 초기화 ──────────────────────────
        ip = rospy.get_param('~franka_ip', '172.16.0.2')
        self.robot   = Robot(ip)
        self.robot.relative_dynamics_factor = 0.10     # 평상시 10 %
        self.gripper = Gripper(ip)

        self.emergency_stop   = False                  # 감속 flag
        self.last_emerg_stamp = None

        # ───── CSV 로그 파일 ────────────────────────
        self.csv_path = os.path.expanduser('~/joint_speed_log.csv')
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ['stamp','j1','j2','j3','j4','j5','j6','j7'])

        # ───── ROS I/O ──────────────────────────────
        rospy.Subscriber('/emergency_stop',      Bool,   self.emerg_cb,  queue_size=1)
        rospy.Subscriber('/emergency_timestamp', Header, self.stamp_cb,  queue_size=1)

        rospy.Subscriber('/op_mani/pose_list',    Float64MultiArray,
                         self.pose_cb,  queue_size=3)
        rospy.Subscriber('/op_mani/pose_gripper', Float64,
                         self.grip_cb,  queue_size=3)

        self.pose_pub = rospy.Publisher('/panda_csv',  Float64MultiArray, queue_size=5)
        self.lat_pub  = rospy.Publisher('/stop_latency', Float64,        queue_size=5)
        self.vel_pub  = rospy.Publisher('/joint_speed', Float64MultiArray, queue_size=5)

        # 20hz joint 저장
        rospy.Timer(rospy.Duration(0.05), self.joint_speed_timer)

    def _read_joint_vel(self):
        js = self.robot.current_joint_state
        if   hasattr(js, 'velocities'): return js.velocities
        elif hasattr(js, 'q_dot'):      return js.q_dot
        elif hasattr(js, 'velocity'):   return js.velocity
        else:                           return [float('nan')]*7

    def _append_csv(self, stamp, vel):
        try:
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([stamp] + list(vel))
        except Exception as e:
            rospy.logerr(f'CSV write error: {e}')
   
    #  callback
  
    def joint_speed_timer(self, _):
        try:
            vel = self._read_joint_vel()
            now = rospy.Time.now().to_sec()
            self._append_csv(now, vel)
            self.vel_pub.publish(Float64MultiArray(data=list(vel)))
        except Exception as e:
            rospy.logwarn_throttle(5.0, f'Joint speed read fail: {e}')

    
    #  Emergency 처리
    
    def stamp_cb(self, msg):
        self.last_emerg_stamp = msg.stamp.to_sec()

    def emerg_cb(self, msg):
        # EMERGENCY TRUE 감속 모드 진입 
        if msg.data and not self.emergency_stop:
            self._handle_emergency()

        # EMERGENCY FALSE 평상시 복구 
        elif not msg.data and self.emergency_stop:
            rospy.loginfo('[ROBOT] Emergency cleared → recovering')
            self.emergency_stop = False
            self._resume_robot()

    # scaling factor 조절
    def _handle_emergency(self):
        rospy.logwarn('[ROBOT] EMERGENCY — speed scale 0.03 (soft slow-down)')
        self.emergency_stop = True
        self.robot.relative_dynamics_factor = 0.03

        
        t_cmd = self.last_emerg_stamp or rospy.Time.now().to_sec()
        self.lat_pub.publish(rospy.Time.now().to_sec() - t_cmd)

    # scaling facotr 복구
    def _resume_robot(self):
        try:
            if hasattr(self.robot, 'recover_from_errors'):
                self.robot.recover_from_errors()
        except Exception as e:
            rospy.logwarn(f'recover_from_errors() failed: {e}')

        self.robot.relative_dynamics_factor = 0.10
        rospy.loginfo('[ROBOT] ready – motion commands accepted')

    #  pose, gripper 동작
    def pose_cb(self, msg):
        if self.emergency_stop:
            pass
        if len(msg.data) == 7:
            try:
                quat   = msg.data[3:7]
                motion = CartesianMotion(
                    Affine(msg.data[0:3], quat), ReferenceType.Absolute)
                self.robot.move(motion, asynchronous=True)
            except Exception as e:
                rospy.logerr(f'Move failed: {e}')
                self.emergency_stop = True  

        # end-effector 위치
        try:
            ee = self.robot.current_cartesian_state.pose.end_effector_pose
            self.pose_pub.publish(Float64MultiArray(
                data=[ee.translation[0], ee.translation[1], ee.translation[2]]))
        except Exception:
            pass

    def grip_cb(self, msg):
        if self.emergency_stop:
            pass
        try:
            width = np.clip(msg.data, 0.01, 0.105)
            self.gripper.move_async(width, 0.05)
        except Exception as e:
            rospy.logerr(f'Gripper move failed: {e}')

if __name__ == '__main__':
    try:
        FrankyOP()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

