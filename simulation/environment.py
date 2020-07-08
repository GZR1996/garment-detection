import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import imageio
from threading import Thread

ASSERT_NAME = 'asserts'
DATA_FOLDER_NAME = 'data'

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
TABLE_PATH = os.path.join(DIRECTORY, ASSERT_NAME, 'table', 'table.urdf')
CLOTH_PATH = os.path.join(DIRECTORY, ASSERT_NAME, 'clothing', 'cloth_z_up.obj')

output_attributes = ['width', 'height', 'rgbPixels', 'depthPixels', 'segmentationMaskBuffer']

# parameters of data collection
RBG_PATH = os.path.join(DIRECTORY, DATA_FOLDER_NAME, 'rgb')
BIN_PATH = os.path.join(DIRECTORY, DATA_FOLDER_NAME, 'bin')
if not os.path.exists(os.path.join(DIRECTORY, DATA_FOLDER_NAME)):
    os.mkdir(os.path.join(DIRECTORY, DATA_FOLDER_NAME))
if not os.path.exists(RBG_PATH):
    os.mkdir(RBG_PATH)
if not os.path.exists(BIN_PATH):
    os.mkdir(BIN_PATH)


def save(file_name, camera_image):
    """
    save file from pybullet data
    :param file_name: name of file to save
    :param camera_image: data from pybullet
    :return:
    """
    rgb_data = np.array(camera_image[2])
    path = os.path.join(RBG_PATH, file_name + '.png')
    imageio.imwrite(path, rgb_data)

    raw_depth_data = np.array(camera_image[3])
    segmentation_data = np.array(camera_image[4])
    depth_data = raw_depth_data.copy()
    # only keep depth data of cloth
    depth_data[(raw_depth_data != 1.0) & (segmentation_data == 0.0)] = 1

    # save data and compress as a npz file
    path = os.path.join(BIN_PATH, file_name)
    np.savez_compressed(path, raw_depth=raw_depth_data, segmentation=segmentation_data, depth=depth_data)


def cal_parameter_group(parameter_count):
    nums = list(str(parameter_count))
    if len(nums) == 1:
        nums.insert(0, 0)
        nums.insert(0, 0)
    elif len(nums) == 2:
        nums.insert(0, 0)
    return [int(n) for n in nums]


class Environment:

    def __init__(self, table_position, cloth_position):
        # parameters of table
        self.table_position = table_position

        # parameters of cloth
        self.cloth_position = cloth_position
        self.elastic_stiffness_range = np.arange(40.0, 140.0, 10.0)
        self.damping_stiffness_range = np.arange(0.1, 1.1, 0.1)
        self.bending_stiffness_range = np.arange(2.0, 22.0, 2.0)

        # parameters of pybullet
        self.physicsClient = p.connect(p.GUI)
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL

        # parameters of camera
        self.available_eye_positions = [[2, 0.0, 1.0], [0.0, 2, 1.0],
                                        [-2, 0.0, 1.0], [0.0, -2.5, 1.0], [0.0, -1.0, 3.0]]
        self.available_target_positions = [[0., -0.55, 1.0], [0., -1.0, 1.0],
                                           [0., -0.85, 1.0], [0., -0.85, 1.0], [0., -0.85, 1.0]]
        self.target_position = np.array(self.cloth_position - [0.0, -0.2, 0.5])
        self.up_vector = np.array([0.0, 0.0, 1.0])
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=3.1)
        self.view_matrics = []
        for eye_position, target_position in zip(self.available_eye_positions, self.available_target_positions):
            view_matrix = p.computeViewMatrix(cameraEyePosition=eye_position,
                                              cameraTargetPosition=target_position,
                                              cameraUpVector=self.up_vector)
            self.view_matrics.append(view_matrix)

        self.height = 256
        self.width = 256

    def simulate(self):
        """
        simulate the environment using pybullet and save file
        :return:
        """
        for parameter_count in range(0, 1000, 1):
            indexes = cal_parameter_group(parameter_count)
            spring_elastic_stiffness = self.elastic_stiffness_range[indexes[0]]
            spring_damping_stiffness = self.damping_stiffness_range[indexes[1]]
            spring_bending_stiffness = self.bending_stiffness_range[indexes[2]]
            table_id, cloth_id, anchor_ids = self.load_world(spring_elastic_stiffness,
                                                             spring_damping_stiffness,
                                                             spring_bending_stiffness)

            iteration = 0
            iteration_name = "{:.1f}_{:.1f}_{:.1f}".format(spring_elastic_stiffness,
                                                           spring_damping_stiffness,
                                                           spring_bending_stiffness)
            print("Saving file: ", iteration_name)
            start = time.time()
            for step in range(1500):
                # suspend and release the cloth
                if step == 0:
                    p.resetBaseVelocity(cloth_id, linearVelocity=[0, 0, 0.5])
                    # p.applyExternalForce(cloth_id, 0, [0, 0, 1000], [0, 0, 0], p.WORLD_FRAME)
                elif step == 500:
                    p.resetBaseVelocity(cloth_id, linearVelocity=[0, 0, 0])
                    for anchor_id in anchor_ids:
                        p.removeConstraint(anchor_id)

                # after releasing the cloth, record the data every 25 frame of simulation
                if step % 50 == 0:
                    pass
                    for eye_position, view_matrix in enumerate(self.view_matrics):
                        camera_image = p.getCameraImage(width=self.width, height=self.height,
                                                        viewMatrix=view_matrix,
                                                        projectionMatrix=self.projection_matrix,
                                                        renderer=self.renderer)
                        file_name = iteration_name + '_' + str(iteration) + '_' + str(eye_position)
                        thread = Thread(target=save, args=(file_name, camera_image))
                        thread.start()
                        time.sleep(0.2)  # force the program sleep to avoid too many threads run at the same time

                    iteration += 1
                    print("finish saving in ", time.time() - start)

                p.stepSimulation()

    def load_world(self, spring_elastic_stiffness, spring_damping_stiffness, spring_bending_stiffness):
        """
        init plane and table of the environment
        :param spring_elastic_stiffness
        :param spring_damping_stiffness
        :param spring_bending_stiffness
        :return:
        """
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # reset the environment to enable simulate deformable object
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW)
        p.setGravity(0, 0, -10)

        table_id = p.loadURDF(TABLE_PATH, basePosition=self.table_position,
                              baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2.0]),
                              flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        # init cloth
        cloth_id = p.loadSoftBody(CLOTH_PATH, basePosition=self.cloth_position,
                                  scale=0.2, mass=1.,
                                  useNeoHookean=0, useBendingSprings=1, useMassSpring=1, useSelfCollision=1,
                                  springElasticStiffness=spring_elastic_stiffness,
                                  springDampingStiffness=spring_damping_stiffness,
                                  springBendingStiffness=spring_bending_stiffness,
                                  springDampingAllDirections=1, frictionCoeff=.5, useFaceContact=1)

        anchor_1 = p.createSoftBodyAnchor(cloth_id, 0, -1, -1)
        anchor_2 = p.createSoftBodyAnchor(cloth_id, 1, -1, -1)

        return table_id, cloth_id, (anchor_1, anchor_2)

    def test(self):
        """
        only for programming test
        :return:
        """
        spring_elastic_stiffness, spring_damping_stiffness, spring_bending_stiffness = cal_parameter_group(555)
        table_id, cloth_id, anchor_ids = self.load_world(spring_elastic_stiffness, spring_damping_stiffness,
                                                         spring_bending_stiffness)

        i = 4
        available_eye_position = [[2, 0.0, 1.0], [0.0, 2, 1.0],
                                  [-2, 0.0, 1.0], [0.0, -2.5, 1.0], [0.0, -1.0, 3.0]]
        available_target_position = [[0., -0.55, 1.0], [0., -1.0, 1.0],
                                     [0., -0.85, 1.0], [0., -0.85, 1.0], [0., -0.85, 1.0]]
        eye_position = available_eye_position[i]
        target_position = available_target_position[i]
        view_matrix = p.computeViewMatrix(cameraEyePosition=eye_position,
                                          cameraTargetPosition=target_position,
                                          cameraUpVector=self.up_vector)
        self.available_target_position = []
        self.target_position = np.array(self.cloth_position - [0.0, -0.2, 0.5])
        self.up_vector = np.array([0.0, 0.0, 1.0])
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=10.1)

        for step in range(10000):
            camera_image = p.getCameraImage(width=self.width, height=self.height,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=self.projection_matrix,
                                            renderer=self.renderer)
            # suspend and release the cloth
            if step < 60:
                p.applyExternalForce(cloth_id, 0, [0, 0, 10], [0, 0, 0], p.WORLD_FRAME)
            # after releasing the cloth, record the data every 25 frame of simulation

            p.stepSimulation()
