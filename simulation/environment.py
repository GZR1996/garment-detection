import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import imageio
from threading import Thread
from PIL import Image
import matplotlib.pyplot as plt

directory = os.path.dirname(os.path.realpath(__file__))
table_path = os.path.join(directory, 'asserts', 'table', 'table.urdf')
cloth_path = os.path.join(directory, 'asserts', 'clothing', 'hospitalgown_reduced.obj')

output_attributes = ['width', 'height', 'rgbPixels', 'depthPixels', 'segmentationMaskBuffer']

# parameters of data collection
rgb_path = os.path.join(directory, 'data', 'rgb')
depth_path = os.path.join(directory, 'data', 'depth')
segmentation_path = os.path.join(directory, 'data', 'segmentation')
if not os.path.exists(rgb_path):
    os.mkdir(rgb_path)
if not os.path.exists(depth_path):
    os.mkdir(depth_path)
if not os.path.exists(segmentation_path):
    os.mkdir(segmentation_path)


def save(file_name, camera_image):
    """
    save file from pybullet data
    :param file_name: name of file to save
    :param camera_image: data from pybullet
    :return:
    """
    rgb_data = np.array(camera_image[2])
    path = os.path.join(rgb_path, file_name + '.png')
    imageio.imwrite(path, rgb_data)

    depth_data = np.array(camera_image[3])
    path = os.path.join(depth_path, file_name + '.npy')
    np.save(path, depth_data)

    segmentation_data = np.array(camera_image[4])
    path = os.path.join(segmentation_path, file_name + '.npy')
    np.save(path, segmentation_data)


class Environment:

    def __init__(self, table_position, table_orientation,
                 cloth_position, cloth_orientation, cloth_color, cloth_line_color):
        # parameters of table
        self.table_position = table_position
        self.table_orientation = table_orientation

        # parameters of cloth
        self.cloth_position = cloth_position
        self.cloth_color = cloth_color
        self.cloth_line_color = cloth_line_color
        self.cloth_indices_boundary = 300

        # parameters of pybullet
        self.physicsClient = p.connect(p.GUI)
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL

        self.available_position = [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [-2.0, 0.0, 1.0],
                                   [0.0, -2.0, 1.0], [0.0, 0.0, 2.0]]
        self.target_position = np.array(self.cloth_position - [0.0, -0.2, 0.5])
        self.up_vector = np.array([0.0, 0.0, 1.0])

        self.height = 256
        self.width = 256
        self.file_count = 0

    def simulate(self):
        """
        simulate the environment using pybullet
        :return:
        """
        num_joints = 1

        for i in range(0, self.cloth_indices_boundary, 10):
            joints = [(i + 10 * offset) % self.cloth_indices_boundary for offset in range(num_joints)]
            table_id, cloth_id, anchor_ids = self.load_world(joints)

            for step in range(350):
                if step == 100:
                    for anchor_id in anchor_ids:
                        p.removeConstraint(anchor_id)
                # p.getCameraImage(self.width, self.height, renderer=self.renderer)
                p.stepSimulation()
                time.sleep(1. / 240.)

            for num, eye_position in enumerate(self.available_position[0:1]):
                view_matrix = p.computeViewMatrix(cameraEyePosition=eye_position,
                                                  cameraTargetPosition=self.target_position,
                                                  cameraUpVector=self.up_vector)
                # view_matrix = p.computeViewMatrix(cameraEyePosition=[0, 2, 1],
                #                                   cameraTargetPosition=[0, 0, 0.5],
                #                                   cameraUpVector=[0, 0, 1])
                projection_matrix = p.computeProjectionMatrixFOV(fov=45.0,
                                                                 aspect=1.0,
                                                                 nearVal=0.1,
                                                                 farVal=3.1)
                camera_image = p.getCameraImage(width=self.width, height=self.height,
                                                viewMatrix=view_matrix, projectionMatrix=projection_matrix,
                                                renderer=self.renderer)
                file_name = "{}_{}".format(self.file_count, num)
                thread = Thread(target=save, args=(file_name, camera_image))
                thread.start()

            self.file_count += 1
            time.sleep(0.05)
            p.resetSimulation()

    def load_world(self, joint_indices):
        """
        init plane and table of the environment
        :param joint_indices: a list of joint indices
        :return:
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        table_id = p.loadURDF(table_path, basePosition=[0, -0.35, 0],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2.0]),
                              flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        # init cloth
        cloth_id = p.loadSoftBody(cloth_path, basePosition=self.cloth_position, scale=1, mass=1., useNeoHookean=0,
                                  useBendingSprings=1, useMassSpring=1, springElasticStiffness=40,
                                  springDampingStiffness=.1, springDampingAllDirections=1,
                                  useSelfCollision=1, frictionCoeff=.5, useFaceContact=1)

        anchors = [p.createSoftBodyAnchor(cloth_id, index, -1, -1) for index in joint_indices]

        return table_id, cloth_id, anchors
