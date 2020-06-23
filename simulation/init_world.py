import numpy as np
from simulation.environment import Environment

# the parameter of cloth
table_position = np.array([0, -0.35, 0])
table_orientation = np.array([0, 0, np.pi / 2.0])

# the parameter of cloth
cloth_position = np.array([-0.03, -0.2, 1])
cloth_orientation = np.array([0, 0, np.pi])
cloth_color = np.array([1, 0.5, 0.5, 1])
cloth_line_color = np.array([0, 0.5, 0, 1])

environment = Environment(table_position, table_orientation,
                          cloth_position, cloth_orientation, cloth_color, cloth_line_color)
environment.simulate()



# import os
# import time
# import numpy as np
# import pybullet as p
# import pybullet_data
#
# robot_type = 'pr2'
# directory = os.path.dirname(os.path.realpath(__file__))
# table_path = os.path.join(directory, 'asserts', 'table', 'table.urdf')
# cloth_path = os.path.join(directory, 'asserts', 'clothing', 'hospitalgown_reduced.obj')
# robot_path = os.path.join(directory, 'asserts', 'PR2', 'pr2_no_torso_lift_tall.urdf')
# print(directory)
#
# physicsClient = p.connect(p.GUI)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
# cubeStartPos = [0,0,0]
# cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# # boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
#
# # table
# furniture_id = p.loadURDF(table_path, basePosition=[0, -0.35, 0],
#                           baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2.0]),
#                           flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
#
# # cloth
# gripper_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 0, 0])
# gripper_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0001)
#
# state = p.getBasePositionAndOrientation(furniture_id)
# start_ee_pos = np.array(state[0])
# cloth_orig_pos = [1, 1, 3]  # np.array([0.34658437, -0.30296362, 1.20023387])
# cloth_offset = start_ee_pos - cloth_orig_pos
#
# cloth_attachment = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=gripper_visual,
#                                      baseCollisionShapeIndex=gripper_collision,
#                                      basePosition=np.array([0, 0, 0]),
#                                      useMaximalCoordinates=1)
#
# cloth_id = p.loadCloth(cloth_path, scale=1.0, mass=0.23,
#                        position=np.array([-0.03, -0.2, 1]),
#                        orientation=p.getQuaternionFromEuler([0, 0, np.pi]),
#                        bodyAnchorId=cloth_attachment,
#                        anchors=[2087, 3879, 3681, 3682, 2086, 2041, 987, 2042, 2088, 1647, 2332],
#                        collisionMargin=0.04, rgbaColor=np.array([1, 0.5, 0.5, 1]),
#                        rgbaLineColor=np.array([0, 0.5, 0, 1]))
#
# print(p.getCameraImage(128, 128))
#
# with open('1.txt', 'w') as fout:
#     fout.write(str(p.getCameraImage(64, 64)))
#
# for i in range (1000000):
#     p.stepSimulation()
#     time.sleep(1./240.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(cloth_id)
# print(cubePos,cubeOrn)
# p.disconnect()
