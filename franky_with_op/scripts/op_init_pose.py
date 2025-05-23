#!/usr/bin/env python3

import rospy
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest, SetKinematicsPoseResponse

def set_kinematics_pose(x, y, z, time):
    rospy.wait_for_service('/op_mani/goal_task_space_path_position_only')
    try:
        set_pose = rospy.ServiceProxy('/op_mani/goal_task_space_path_position_only', SetKinematicsPose)
        
        request = SetKinematicsPoseRequest()
        request.planning_group = "arm"
        request.end_effector_name = "gripper"
        request.kinematics_pose.pose.position.x = x
        request.kinematics_pose.pose.position.y = y
        request.kinematics_pose.pose.position.z = z
        request.kinematics_pose.pose.orientation.x = 0.0
        request.kinematics_pose.pose.orientation.y = 0.707
        request.kinematics_pose.pose.orientation.z = 0.0
        request.kinematics_pose.pose.orientation.w = 0.707
        request.kinematics_pose.max_accelerations_scaling_factor = 0.5
        request.kinematics_pose.max_velocity_scaling_factor = 0.5
        request.kinematics_pose.tolerance = 0.01
        request.path_time = time
        
        response = set_pose(request)

        if response.is_planned:
            rospy.loginfo("Kinematics pose set successfully")
        else:
            rospy.logwarn("Failed to set kinematics pose")
    
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

if __name__ == '__main__':
    rospy.init_node('init_op_pose')

    coordinates = [0.1333, 0.0, 0.11444]

    try:
        #set_kinematics_pose(0.13365, 0.0, 0.11475, 2.0)
        set_kinematics_pose(coordinates[0], coordinates[1], coordinates[2], 2.0)

    except rospy.ROSInterruptException:
        pass