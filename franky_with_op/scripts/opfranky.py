#!/usr/bin/env python3

import rospy, actionlib, sys

import numpy as np
from franky import Affine, CartesianMotion, Robot, ReferenceType, Gripper
from scipy.spatial.transform import Rotation

from std_msgs.msg import Float64MultiArray, Float64

class franky_op:

    def __init__(self):
        
        rospy.init_node("franky_op")
        rate = rospy.Rate(100)

        self.robot = Robot("172.16.0.2")

        # Velocity, Acceleration, Jerk to 5%
        self.robot.relative_dynamics_factor = 0.1
        self.state = self.robot.state

        self.gripper = Gripper("172.16.0.2")
        self.speed = 0.05  # [m/s]
        self.force = 20.0

        # print('\nPose: ', self.robot.current_pose)
        # print('\nO_TT_E: ', state.O_T_EE)

        # test
        # topic -> data -> subtract -> print -> how long distance will be moved
        # relative and absolute, relative => limit => if jerk -> 50%

        sub_pose = rospy.Subscriber('/op_mani/pose_list', Float64MultiArray, self.cb, queue_size=3)
        sub_gripper = rospy.Subscriber('/op_mani/pose_gripper', Float64, self.cb2, queue_size=3)
        # absolute pose
        #motion = CartesianMotion(Affine([0.0, 0.0, -0.1]), ReferenceType.Absolute)
        
        # real-time => asynchronous=True
        #robot.move(motion, asynchronous=True)

        # publish panda robot pose

        
        self.pub = rospy.Publisher('/panda_csv', Float64MultiArray, queue_size= 10)


    def cb(self, data):
        if len(data.data) == 7:
            #print("\ndata received")
            #print(data.data)

            quat = np.array([data.data[3],data.data[4],data.data[5],data.data[6]])
            motion = CartesianMotion(Affine([data.data[0], data.data[1], data.data[2]], quat), ReferenceType.Absolute)

            # robot move
            self.robot.move(motion, asynchronous=True)

        else :
            print("wrong data")

        
        # csv pub
        #print(self.ee_pose.translation[0])
        
        cartesian_state = self.robot.current_cartesian_state
        robot_pose = cartesian_state.pose
        self.ee_pose = robot_pose.end_effector_pose

        panda_pose = Float64MultiArray()

        #panda_pose.data = [self.state.O_T_EE[12], self.state.O_T_EE[13], self.state.O_T_EE[14]]
        panda_pose.data = [self.ee_pose.translation[0], self.ee_pose.translation[1], self.ee_pose.translation[2]]
        print(panda_pose)
        self.pub.publish(panda_pose)

        rospy.sleep(0.001)

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