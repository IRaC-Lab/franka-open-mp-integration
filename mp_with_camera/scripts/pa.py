import rospy
import tf
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Initialize the ROS node
rospy.init_node('marker_position_example')

# Set up a TF listener
listener = tf.TransformListener()

# Wait for the TF transform to become available
listener.waitForTransform("panda_link0", "panda_hand", rospy.Time(), rospy.Duration(1.0))

# Get the transform between the frames
try:
    (trans, rot) = listener.lookupTransform("panda_link0", "sphere_namespace", rospy.Time(0))
    
    # Transform the marker's position from panda_link0 to the target frame (sphere_namespace)
    transformed_position = listener.transformPoint("panda_link0", human_wrist.pose.position)

    # Print the transformed position
    rospy.loginfo("Transformed Position: x=%.2f, y=%.2f, z=%.2f",
                  transformed_position.x, transformed_position.y, transformed_position.z)
except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
    rospy.logerr("Error during transformation: %s", str(e))
