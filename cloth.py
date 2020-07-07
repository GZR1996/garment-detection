import pybullet as p
from time import sleep
import imageio
import os

directory = os.path.dirname(os.path.realpath(__file__))
cloth_path = os.path.join(directory, 'simulation', 'asserts', 'clothing', 'hospitalgown_reduced.obj')

physicsClient = p.connect(p.GUI)
import pybullet_data

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW)

gravZ = -10
p.setGravity(0, 0, gravZ)

planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
planeId = p.loadURDF("plane.urdf", [0, 0, -2], planeOrn)

boxId = p.loadURDF("cube.urdf", [0, 1, 2], useMaximalCoordinates=True)

clothId = p.loadSoftBody("cloth_z_up.obj", basePosition=[0, 0, 2], scale=0.5, mass=1., useNeoHookean=0,
                         useBendingSprings=1, useMassSpring=1, springElasticStiffness=40, springDampingStiffness=.99,
                         springDampingAllDirections=1, useSelfCollision=0, frictionCoeff=.5, useFaceContact=1)
p.resetBaseVelocity(clothId, linearVelocity=[0, 0, 1])
p.createSoftBodyAnchor(clothId, 0, -1, -1)
# p.createSoftBodyAnchor(clothId, 1, -1, -1)
# p.createSoftBodyAnchor(clothId, 3, boxId, -1, [0.5, -0.5, 0])
# p.createSoftBodyAnchor(clothId, 2, boxId, -1, [-0.5, -0.5, 0])
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(1)

t = True
i = 0
while p.isConnected():
    p.applyExternalForce(clothId, 3, [0, 0, 1000], [0, 0, 0], p.WORLD_FRAME)
    i += 1
    p.setGravity(0, 0, gravZ)
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[0, 2, 0],
        cameraTargetPosition=[0, 0, 0.5],
        cameraUpVector=[0, 0, 1])
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=10)
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=224,
        height=224,
        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    sleep(1. / 240.)
