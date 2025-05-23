#!/usr/bin/env python3

import rospy, actionlib, sys, math

import numpy as np
from franky import Affine, CartesianMotion, Robot, ReferenceType
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray

class franky_test:

    def __init__(self):
        
        rospy.init_node("franky_test")

        robot = Robot("172.16.0.2")

        # Velocity, Acceleration, Jerk to 5%
        robot.relative_dynamics_factor = 0.05

        #quat = np.array([0.9999914701616789, -0.0015339689491569357, 0.0038349065883222905, 5.88267780769409e-06])
        
        # absolute pose
        #motion = CartesianMotion(Affine([0.3166817, -0.000887, 0.4316879], quat), ReferenceType.Absolute)
        
        # real-time => asynchronous=True
        #robot.move(motion)
        
        state = robot.state
        
        print(state.O_T_EE)

if __name__ == '__main__':

        try:
            manipulator_t = franky_test()
        except rospy.ROSInterruptException:
            pass
