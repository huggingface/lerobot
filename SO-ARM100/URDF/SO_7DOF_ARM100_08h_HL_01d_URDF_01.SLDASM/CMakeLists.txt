cmake_minimum_required(VERSION 2.8.3)

project(SO_7DOF_ARM100_08h_HL_01d_URDF_01.SLDASM)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
