#!/usr/bin/env python3

import sys, select, termios, tty
import rospy
import numpy as np
import csv
import time

from std_msgs.msg import Float64MultiArray
from open_manipulator_msgs.msg import KinematicsPose
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from scipy.spatial.transform import Rotation as R
from math import cos, sin, radians


class EEPoseSave:
    
    def __init__(self):

        rospy.init_node('save_end_effector_pose', anonymous=True)

        rate = rospy.Rate(100)

        self.save_sub1 = rospy.Subscriber('/panda_csv', Float64MultiArray, self.cb, queue_size=10)
        
        self.save_sub2 = rospy.Subscriber('/op_mani/gripper/kinematics_pose', KinematicsPose, self.cb2, queue_size=10)



    def cb(self, data):

        EE_POSE = data.data

        with open('EEPOSE_panda.csv', 'a', newline='') as csvfile:
            fieldnames = ['x', 'y', 'z']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:  # Write header if the file is empty
                writer.writeheader()

            writer.writerow({'x': EE_POSE[0],
                            'y': EE_POSE[1],
                            'z': EE_POSE[2],
                            })

            time.sleep(0.001)

    def cb2(self, data):

        EE_POSE = [data.pose.position.x, data.pose.position.y, data.pose.position.z]

        with open('EEPOSE_op.csv', 'a', newline='') as csvfile:
            fieldnames = ['x', 'y', 'z']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:  # Write header if the file is empty
                writer.writeheader()

            writer.writerow({'x': EE_POSE[0],
                            'y': EE_POSE[1],
                            'z': EE_POSE[2],
                            })

            time.sleep(0.001)
                
if __name__ == "__main__":

    EEPoseSave()

    rospy.spin()
