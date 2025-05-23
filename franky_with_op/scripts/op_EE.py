#!/usr/bin/env python3

import sys, select, termios, tty
import rospy
import actionlib
import numpy as np

from open_manipulator_msgs.msg import KinematicsPose

from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import InteractiveMarkerFeedback, Marker, InteractiveMarkerControl
from geometry_msgs.msg import PoseStamped, Quaternion, Point

from franka_msgs.msg import FrankaState

from scipy.spatial.transform import Rotation as R
from math import cos, sin, radians
from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveActionGoal

from std_msgs.msg import Float64MultiArray, Float64

# Check Torque
from open_manipulator_msgs.msg import OpenManipulatorState

class Comb_Manipulator:

    def __init__(self):

        rospy.init_node('opmani_to_panda_publisher')

        rate = rospy.Rate(1000)

        # Wait until Init Open Manipulator Pose
        global result

        # waiting for toggle torque off
        act_sub = rospy.Subscriber('/op_mani/states', OpenManipulatorState, self.cb3)

        rospy.sleep(1)

        while not rospy.is_shutdown() :
            
            ##### \"ACTUATOR_DISABLED\" <==== \"
            if result == "\"ACTUATOR_DISABLED\"":
                break

            # else:
            #     print('\nresult: ', result)
            #     print('waiting for torque off...')


        # Get Open Manipulator Pose data
        get_op_pose = rospy.Subscriber('/op_mani/gripper/kinematics_pose', KinematicsPose, self.cb, queue_size=3)
        get_gipper = rospy.Subscriber('/op_mani/joint_states', JointState, self.cb2, queue_size=10)

        # Publish data to Panda
        self.grip_pub = rospy.Publisher('/op_mani/pose_gripper', Float64 , queue_size= 5)
        self.pub = rospy.Publisher('/op_mani/pose_list', Float64MultiArray, queue_size=3)

    def cb(self, data):

        # data => EEF Postion of Open Manipulator X

        # X Y Z
        pose_x = round(data.pose.position.x*2.25, 7)
        pose_y = round(data.pose.position.y*2.25, 7)
        pose_z = round(data.pose.position.z*4.793 - 0.063, 7)

        # Quat
        x1_ori = data.pose.orientation.x
        y1_ori = data.pose.orientation.y
        z1_ori = data.pose.orientation.z
        w1_ori = data.pose.orientation.w

        # Rotation Matrix
        r = R.from_quat([x1_ori, y1_ori, z1_ori, w1_ori])   
        r1 = r.as_matrix()
        A = np.array([[0,0,1],[0,-1,0],[1,0,0]])

        matrix = np.dot(r1,A)

        B = R.from_matrix(matrix)

        # quat = panda robot quat
        quat = B.as_quat()

        # data for topic msg
        pose_list = Float64MultiArray()

        # send 7 data
        pose_list.data = [pose_x, pose_y, pose_z, quat[0], quat[1], quat[2], quat[3]]  # pose data

        #print("\n",data.pose.position.x)
        #print("\nsended Pose Data: ", pose_list.data)

        self.pub.publish(pose_list)

        rospy.sleep(0.001)        

    
    def cb2(self, data):
         
        j = data.position
        j_op = j[4]

        # 0.058 is close and 0.028 is open
        
        ## re calculate!!!!
        
        panda_grip = ((-3*j_op) + 0.184)

        self.grip_pub.publish(panda_grip)
                
    def cb3(self, data):
        global result

        result = data.open_manipulator_actuator_state
         

if __name__ == '__main__':

        manipulator = Comb_Manipulator()          
        rospy.spin()
