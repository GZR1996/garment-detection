import numpy as np
from simulation.environment import Environment

# the parameter of cloth
table_position = np.array([0, -0.35, 0])
table_orientation = np.array([0, 0, np.pi / 2.0])

# the parameter of cloth
cloth_position = np.array([0, -0.75, 0.65])
cloth_orientation = np.array([0, 0, np.pi])
cloth_color = np.array([1, 0.5, 0.5, 1])
cloth_line_color = np.array([0, 0.5, 0, 1])

environment = Environment(table_position, table_orientation,
                          cloth_position, cloth_orientation, cloth_color, cloth_line_color)
environment.simulate()
# environment.test()