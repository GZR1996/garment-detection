import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import imageio

directory = os.path.dirname(os.path.realpath(__file__))
table_path = os.path.join(directory, 'asserts', 'table', 'table.urdf')
cloth_path = os.path.join(directory, 'asserts', 'clothing', 'hospitalgown_reduced.obj')

output_attributes = ['width', 'height', 'rgbPixels', 'depthPixels', 'segmentationMaskBuffer']


class Environment:

    def __init__(self, table_position, table_orientation,
                 cloth_position, cloth_orientation, cloth_color, cloth_line_color):
        # parameters of table
        self.table_position = table_position
        self.table_orientation = table_orientation

        # parameters of cloth
        self.cloth_position = cloth_position
        self.cloth_orientation = cloth_orientation
        self.cloth_color = cloth_color
        self.cloth_line_color = cloth_line_color

        # parameters of pybullet
        self.physicsClient = p.connect(p.GUI)

        self.eye_position = np.array([2.0, 2.0, 2.0])
        self.target_position = np.array(self.cloth_position)
        self.up_vector = np.array([0.0, 0.0, 1.0])
        self.viewMatrix = p.computeViewMatrix(self.eye_position, self.target_position, self.up_vector)

        # parameters of data collection
        self.height = 1920
        self.width = 1080
        self.count = 0
        self.rgb_path = os.path.join(directory, 'data', 'rgb')
        self.depth_path = os.path.join(directory, 'data', 'depth')
        self.segmentation_path = os.path.join(directory, 'data', 'segmentation')
        if not os.path.exists(self.rgb_path):
            os.mkdir(self.rgb_path)
        if not os.path.exists(self.depth_path):
            os.mkdir(self.depth_path)
        if not os.path.exists(self.segmentation_path):
            os.mkdir(self.segmentation_path)

    def simulate(self):
        """
        simulate the environment using pybullet
        :return:
        """
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # init plane and table of the environment
        plane_id = p.loadURDF("plane.urdf")
        table_id = p.loadURDF(table_path, basePosition=[0, -0.35, 0],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2.0]),
                              flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        # init cloth
        gripper_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 0, 0, 0])
        gripper_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.0001)

        cloth_attachment = p.createMultiBody(baseMass=0.2, baseVisualShapeIndex=gripper_visual,
                                             baseCollisionShapeIndex=gripper_collision,
                                             basePosition=np.array([0, 0, 0]),
                                             useMaximalCoordinates=1)

        cloth_id = p.loadCloth(cloth_path, scale=1.0, mass=0.23,
                               position=np.array([-0.03, -0.2, 1]),
                               orientation=p.getQuaternionFromEuler([0, 0, np.pi]),
                               bodyAnchorId=cloth_attachment,
                               anchors=[2087, 3879, 3681, 3682, 2086, 2041, 987, 2042, 2088, 1647, 2332],
                               collisionMargin=0.04, rgbaColor=np.array([1, 0.5, 0.5, 1]),
                               rgbaLineColor=np.array([0, 0.5, 0, 1]))

        # save data
        # camera_distance = 4
        # pitch = -10.0
        # roll = 0
        # up_axis_index = 2
        # t = True
        # i = 0
        # while (p.isConnected()):
        #     for yaw in range(0, 360, 10):
        #         start = time.time()
        #         p.stepSimulation()
        #         stop = time.time()
        #         i += 1
        #         print("stepSimulation %f" % (stop - start))
        #
        #         viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.target_position, camera_distance, yaw, pitch, roll,
        #                                                          up_axis_index)
        #         projectionMatrix = [
        #             1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        #             -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
        #         ]
        #
        #         start = time.time()
        #         img_arr = p.getCameraImage(self.width,
        #                                    self.height,
        #                                    viewMatrix=viewMatrix,
        #                                    projectionMatrix=projectionMatrix,
        #                                    shadow=1,
        #                                    lightDirection=[1, 1, 1])
        #         if t:
        #             t = False
        #             self.save(img_arr)
        #         stop = time.time()
        #         print("renderImage %f" % (stop - start))

        self.save(p.getCameraImage(self.width, self.height, renderer=p.ER_TINY_RENDERER))
        for i in range(1000000):
            p.stepSimulation()
            time.sleep(1. / 240.)

        p.disconnect()

    def save(self, camera_image):
        """
        save file from pybullet data
        :param file_name:
        :param camera_image: data from pybullet
        :return:
        """
        for i, data in enumerate(camera_image[2:]):
            data = np.array(data)
            if i == 0:
                file_name = os.path.join(self.rgb_path, str(self.count)+'.png')
                imageio.imwrite(file_name, data)
            elif i == 1:
                file_name = os.path.join(self.depth_path, str(self.count)+'.txt')
                np.savetxt(file_name, data)
            elif i == 2:
                file_name = os.path.join(self.segmentation_path, str(self.count)+'.txt')
                np.savetxt(file_name, data)

        with open(file_name, 'w') as output_stream:
            outputs = []
            for i, line in enumerate(camera_image[2:]):
                outputs.append('## {} \n {} \n\n'.format(output_attributes[i], str(line).strip()))
            output_stream.writelines(outputs)
