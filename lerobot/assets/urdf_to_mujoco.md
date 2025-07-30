
# Converting an URDF to a MJCF XML for MuJoCo

1. Mujoco does not support `.dae` mesh files.If your mesh files are in `.dae` format, convert them to `.stl` of `.obj` using `meshlab` for example. One way to do this is : `meshlabserver -i in.obj -o out.obj -m vn`. Then rename the paths in the `urdf` file to point to the new mesh files.

2. In the urdf, add the mujoco compiler tag, for example:
```xml
<robot>
	<mujoco>
    	<compiler meshdir="./mujoco_meshes"/>
        <!-- if you get : "Error: inertia must satisfy A + B >= C; use 'balanceinertia' to fix", add balanceinertia="true" to the compiler tag -->
        <!-- You can also add discardvisual="true" discard model elements that are purely visual and have no effect on the simulation. -->
	</mujoco>
    ...
</robot>
```
3. Download MuJoCo binaries https://github.com/google-deepmind/mujoco/releases

4. Convert `urdf` to `mjcf`, Run : `./<path_to_mujoco>/bin/compile robot.urdf robot.xml`

5. In `robot.xml`, define actuators for each joints, for example:

```xml
<mujo>
    ...
    <worldbody>
    ...
    </worldbody>
    <actuator>
        <!-- for position controlled actuators -->
        <position name="yaw" joint="yaw" inheritrange="1"/>
        <!-- for torque controlled actuators -->
        <motor name="yaw" joint="yaw" ctrlrange="-x(N) +y(N)">
        <!-- for velocity controlled actuators -->
        <velocity name="yaw" joint="yaw" ctrlrange="-x(rad/s) +y(rad/s)">
        ...
    </actuator>
</mujoco>
```

(`inheritrange="1"` assumes that the `range` attribute of each joints is correctly defined when controlling in position)
(Add relevant actuator parameters for your robot. Check https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator)

6. If your robot is not fixed in the world, you need to define a `freejoint` in the `worldbody`. For example :
```xml
<worldbody>
    <body name="base" pos="0 0 0">
      <freejoint />
      <body name="yaw" ...>
      </body>
      ...
    </body>
</worldbody>
```

7. You may have to tune some joint parameters in `robot.xml`, mainly `damping`, `frictionloss`, `kp` and `forcerange`. To attribute default values to all joints, you can use the `default` tag. For example:
```xml
<mujoco>
    <default>
        <joint damping="0.2" frictionloss="0.1"/>
        <position kp="10" forcerange="-1.0 1.0"/>
    </default>
    ...
</mujoco>
```

8. Check that everything is working : `./<path_to_mujoco>/bin/simulate robot.xml`. You should be able to control the joints.

9. You can then use your robot in a scene, here is an example scene with lighting, a skybox and a groundplane: 
```xml
<mujoco model="scene">
	<include file="robot.xml" />
	<visual>
		<headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
		<rgba haze="0.15 0.25 0.35 1" />
		<global azimuth="150" elevation="-20" />
	</visual>
	<asset>
		<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
		<texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
		<material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
	</asset>
	<worldbody>
		<light pos="0 0 3" dir="0 0 -1" directional="false" />
		<body name="floor">
			<geom pos="0 0 0" name="floor" size="0 0 .125" type="plane" material="groundplane"/>
		</body>
	</worldbody>
</mujoco>
```

10. Just run `./<path_to_mujoco>/bin/simulate scene.xml` to see your robot in the scene.