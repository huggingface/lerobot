# LeSpectrobot

Little guide to use the DragonTactile robot with the IOLITE-X data acquisition system. First, make sure to have the IOLITE-X connected and configured properly. Then, you can use the `SOFollowerDragonTactile` class to interface with the robot and acquire data from the tactile sensors.

The commande to run teleoperation, training,or evaluation is the same as for the other SO-ARM, just change the robot name to `so_follower_dragontactile`. For example, to run teleoperation:  

```bash
lerobot-teleoperate \
  --robot.type=so101_follower_dragontactile \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ top: {type: opencv, index_or_path: \"/dev/video0\", width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: \"/dev/video2\", width: 640, height: 480, fps: 15} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm
```

TODO list :
- images en float32
- flags dans ACT pour spécifier quelles features prendre en compte pour l'entrainement et l'inférence
- deuxième backbone pour inférence tactile uniquement ( pas préentrainé resnet)
- pour modifier info.json : dataset_writer.py, l.504