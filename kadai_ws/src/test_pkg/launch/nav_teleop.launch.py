import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    
    pkg_share_dir = get_package_share_directory('test_pkg')
    model_path = os.path.join(pkg_share_dir, "models")

    # ignition gazeboがモデルにアクセスできるように設定
    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[os.path.join("/opt/ros/humble", "share"), ":" + model_path]
    )

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world_name', default='world')

    # ignition gazeboの起動設定
    ign_gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_ign_gazebo'), 'launch', 'ign_gazebo.launch.py')
        ]),
        # 【修正】worlds フォルダのなかの world.sdf を見に行くように指定します
        launch_arguments=[('ign_args', [' -r -v 3 ' + os.path.join(pkg_share_dir, "worlds", "world.sdf")])]
    )

    # ros_ign_bridgeの起動設定
    bridge = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            # Velocity command (ROS2 -> IGN)
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/model/demoRobot/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        ],
        remappings=[
            ("/odom/tf", "tf"),
        ],
        output='screen'
    )
    
    teleop_node = Node(
        package='test_pkg',
        executable='teleop_node',
        output='screen',
        # 別ターミナルで起動する設定
        prefix="xterm -e"
    )
    
    return LaunchDescription([
        ign_resource_path,
        ign_gz,
                             
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='If true, use simulated clock'),

        DeclareLaunchArgument(
            'world_name',
            default_value=world_name,
            description='World name'),

        bridge,
        teleop_node,
    ])

