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
final_depth_path = os.path.join(directory, 'data', 'final_depth')
if not os.path.exists(os.path.join(directory, 'data')):
    os.mkdir(os.path.join(directory, 'data'))
if not os.path.exists(rgb_path):
    os.mkdir(rgb_path)
if not os.path.exists(depth_path):
    os.mkdir(depth_path)
if not os.path.exists(segmentation_path):
    os.mkdir(segmentation_path)
if not os.path.exists(final_depth_path):
    os.mkdir(final_depth_path)


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

    final_depth_data = depth_data.copy()
    # only keep depth data of cloth
    path = os.path.join(final_depth_path, file_name + '.npy')
    final_depth_data[(depth_data != 1.0) & (segmentation_data == 0.0)] = 1
    np.save(path, final_depth_data)


def cal_parameter_group(parameter_count):
    nums = list(str(parameter_count))
    if len(nums) == 1:
        nums.insert(0, 0)
        nums.insert(0, 0)
    elif len(nums) == 2:
        nums.insert(0, 0)
    return (int(n)/10 for n in nums)


class Environment:

    def __init__(self, table_position, table_orientation,
                 cloth_position, cloth_orientation, cloth_color, cloth_line_color,
                 num_joints=[1, 2, 3]):
        # parameters of table
        self.table_position = table_position
        self.table_orientation = table_orientation

        # parameters of cloth
        self.cloth_position = cloth_position
        self.cloth_color = cloth_color
        self.cloth_line_color = cloth_line_color
        self.cloth_indices_boundary = 200

        # physics settings
        self.num_joints = num_joints

        # parameters of pybullet
        self.physicsClient = p.connect(p.GUI)
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL

        # parameters of camera
        self.available_position = [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [-2.0, 0.0, 1.0],
                                   [0.0, -2.0, 1.0], [0.0, 0.0, 2.0]]
        self.target_position = np.array(self.cloth_position - [0.0, -0.2, 0.5])
        self.up_vector = np.array([0.0, 0.0, 1.0])
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=3.1)
        self.view_matrics = []
        for eye_position in self.available_position:
            view_matrix = p.computeViewMatrix(cameraEyePosition=eye_position,
                                              cameraTargetPosition=self.target_position,
                                              cameraUpVector=self.up_vector)
            self.view_matrics.append(view_matrix)

        self.height = 256
        self.width = 256

    def set_num_joints(self, num_joints):
        self.num_joints = num_joints

    def simulate(self):
        """
        simulate the environment using pybullet and save file
        file_name: %d(springElasticStiffness)_%d(springDampingStiffness)_%d(springBendingStiffness)_%d(pointsToHold)_%d(holdAnchorIndex)_%(iteration)_%d(eyePosition)
        :return:
        """
        for parameter_count in range(1000):
            spring_elastic_stiffness, spring_damping_stiffness, spring_bending_stiffness = cal_parameter_group(
                parameter_count)
            for num_joint in self.num_joints:
                for i in range(0, self.cloth_indices_boundary, 100):
                    joints = [(i + num_joint * 10 * offset) % self.cloth_indices_boundary
                              for offset in range(num_joint)]
                    table_id, cloth_id, anchor_ids = self.load_world(joints, spring_elastic_stiffness,
                                                                     spring_damping_stiffness, spring_bending_stiffness)

                    iteration = 0
                    iteration_name = "{}_{}_{}_{}_{}".format(spring_elastic_stiffness,
                                                        spring_damping_stiffness,
                                                        spring_bending_stiffness,
                                                        num_joint, joints[0])
                    print("Saving file: ", iteration_name)
                    start = time.time()
                    for step in range(301):
                        # release the cloth
                        if step == 50:
                            for anchor_id in anchor_ids:
                                p.removeConstraint(anchor_id)
                        # after releasing the cloth, record the data every 50 frame of simulation
                        elif step > 50 and step % 50 == 0:
                            for eye_position, view_matrix in enumerate(self.view_matrics):
                                camera_image = p.getCameraImage(width=self.width, height=self.height,
                                                                viewMatrix=view_matrix,
                                                                projectionMatrix=self.projection_matrix,
                                                                renderer=self.renderer)
                                file_name = iteration_name + '_' + str(iteration) + '_' + str(eye_position)
                                thread = Thread(target=save, args=(file_name, camera_image))
                                thread.start()
                                time.sleep(0.005)  # force the program sleep to avoid too many threads run at the same time

                            iteration += 1
                            print("finish saving in ", time.time() - start)

                        p.stepSimulation()


                    p.resetSimulation()  # reset the environment

    def load_world(self, joint_indices, spring_elastic_stiffness, spring_damping_stiffness, spring_bending_stiffness):
        """
        init plane and table of the environment
        :param joint_indices: a list of joint indices
        :param spring_elastic_stiffness
        :param spring_damping_stiffness
        :param spring_bending_stiffness
        :return:
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        plane_id = 0  # p.loadURDF('plane.urdf')

        table_id = p.loadURDF(table_path, basePosition=[0, -0.75, 0],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2.0]),
                              flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        # init cloth
        cloth_id = p.loadSoftBody(cloth_path, basePosition=self.cloth_position, scale=1, mass=1., useNeoHookean=0,
                                  useBendingSprings=1, useMassSpring=1,
                                  springElasticStiffness=spring_elastic_stiffness,
                                  springDampingStiffness=spring_damping_stiffness,
                                  springBendingStiffness=spring_bending_stiffness,
                                  springDampingAllDirections=1,
                                  useSelfCollision=1, frictionCoeff=.5, useFaceContact=1)

        # apply a force to hold cloth with anchors
        anchors = [p.createSoftBodyAnchor(cloth_id, index, -1, -1) for index in joint_indices]

        return table_id, cloth_id, anchors
