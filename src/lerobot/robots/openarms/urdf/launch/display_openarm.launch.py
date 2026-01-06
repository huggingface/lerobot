# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def robot_state_publisher_spawner(context: LaunchContext, arm_type, ee_type, bimanual):
    arm_type_str = context.perform_substitution(arm_type)
    ee_type_str = context.perform_substitution(ee_type)
    bimanual_str = context.perform_substitution(bimanual)

    xacro_path = os.path.join(
        get_package_share_directory("openarm_description"),
        "urdf", "robot", f"{arm_type_str}.urdf.xacro"
    )

    robot_description = xacro.process_file(
        xacro_path,
        mappings={
            "arm_type": arm_type_str,
            "ee_type": ee_type_str,
            "bimanual": bimanual_str,
        }
    ).toprettyxml(indent="  ")

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        )
    ]


def rviz_spawner(context: LaunchContext, bimanual):
    bimanual_str = context.perform_substitution(bimanual)

    rviz_config_file = "bimanual.rviz" if bimanual_str.lower() == "true" else "arm_only.rviz"
    rviz_config_path = os.path.join(
        get_package_share_directory("openarm_description"),
        "rviz", rviz_config_file
    )

    return [
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["--display-config", rviz_config_path],
            output="screen"
        ),
    ]


def generate_launch_description():
    arm_type_arg = DeclareLaunchArgument(
        "arm_type",
        description="Type of arm to visualize (e.g., v10)"
    )

    ee_type_arg = DeclareLaunchArgument(
        "ee_type",
        default_value="openarm_hand",
        description="Type of end-effector to attach (e.g., openarm_hand or none)"
    )

    bimanual_arg = DeclareLaunchArgument(
        "bimanual",
        default_value="false",
        description="Whether to use bimanual configuration"
    )

    arm_type = LaunchConfiguration("arm_type")
    ee_type = LaunchConfiguration("ee_type")
    bimanual = LaunchConfiguration("bimanual")

    robot_state_publisher_loader = OpaqueFunction(
        function=robot_state_publisher_spawner,
        args=[arm_type, ee_type, bimanual]
    )

    rviz_loader = OpaqueFunction(
        function=rviz_spawner,
        args=[bimanual]
    )

    return LaunchDescription([
        arm_type_arg,
        ee_type_arg,
        bimanual_arg,
        robot_state_publisher_loader,
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui"
        ),
        rviz_loader,
    ])
