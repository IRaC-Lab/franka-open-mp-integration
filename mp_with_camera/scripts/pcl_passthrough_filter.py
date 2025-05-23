#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
import pcl.pcl_visualization
import numpy as np
import std_msgs.msg

class PointCloudFilter:
    def __init__(self):
        rospy.init_node('pcl_passthrough_filter', anonymous=True)
        self.sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.callback)
        self.pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=1)
        
    def callback(self, data):
        # Convert ROS PointCloud2 message to NumPy array
        cloud_np = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)))

        # Create a PCL PointCloud object
        cloud = pcl.PointCloud()
        cloud.from_array(cloud_np.astype(np.float32))

        # Apply PassThrough filter
        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name('z')
        passthrough.set_filter_limits(0.5, 3.0)  # Set the filter limits as needed
        cloud_filtered = passthrough.filter()

        # Convert back to ROS PointCloud2 message
        cloud_filtered_np = np.array(cloud_filtered, dtype=np.float32)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = data.header.frame_id
        ros_cloud_filtered = pc2.create_cloud_xyz32(header, cloud_filtered_np)
        
        self.pub.publish(ros_cloud_filtered)

if __name__ == '__main__':
    try:
        PointCloudFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


