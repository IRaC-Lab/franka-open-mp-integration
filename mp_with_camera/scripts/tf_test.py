#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import tf.transformations as tft

def tf_listener():
    # Initialize the ROS node
    rospy.init_node('tf_listener_node', anonymous=True)

    # Create a TF Buffer and a listener
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(10.0)  # 10 Hz
    while not rospy.is_shutdown():
        try:
            # Get the transform between the frames
            trans = tf_buffer.lookup_transform('panda_link6', 'panda_link7', rospy.Time(0), rospy.Duration(1.0))

            # Print out the transformation data
            rospy.loginfo("Translation: x={}, y={}, z={}".format(trans.transform.translation.x,
                                                                  trans.transform.translation.y,
                                                                  trans.transform.translation.z))
            rospy.loginfo("Rotation: x={}, y={}, z={}, w={}".format(trans.transform.rotation.x,
                                                                    trans.transform.rotation.y,
                                                                    trans.transform.rotation.z,
                                                                    trans.transform.rotation.w))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("Transform not available")

        rate.sleep()

if __name__ == '__main__':
    try:
        tf_listener()
    except rospy.ROSInterruptException:
        pass
