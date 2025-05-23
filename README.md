This package is Open Manipulator - Franka Emika Panda Teleoperation System

Check List
1. Connect Open Manipulator and Franka Emika Panda Robot

1-1. Caution, YOU MUST CHECK Franka Emika Panda Robot Init setting -> Ubuntu RT Kernal, Panda Pose Initialize

2. type below code in terminal

roslaunch franky_with_op franky_with_op.launch

3. Turn off Actuator carefully (Use Op Mani control GUI)

4. Two Manipulator Pose is saved in CSV file automatically.

5. visualize on Rviz

roslaunch franka_visualization franka_visualization.launch robot_ip:=172.16.0.2 load_gripper:=true robot:=panda

  <!-- Rviz -->
  <arg name="robot_ip" default="172.16.0.2"/>
  <arg name="load_gripper" default="true"/>
  <arg name="robot" default="panda"/>

  <include file="$(find franka_visualization)/launch/franka_visualization.launch" pass_all_args="true"/>

6. TF Tree Check

How to RUN

1. Install dependency pkg (mediapipe, op-manipulaotr-x, ...)

2. Run franky_with_op.launch, pcl_filter.launch in order

Caution

The manipulator stop related files are missing. So, you would make your own node.

Adapt the code to find the normal vector to your system. Otherwise, errors will occur.
