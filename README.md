# garment-detection

### Simulation

#### Data collection plan
The data collection plan is to collect rgb data, raw depth data, segmentation data and depth data in the simulation with
different physics parameters combination. The simulation is setting different physics of cloth(soft body), suspending it 
from the table and let it fall on the table. The parameters that is available for cloth in this experiment are elastic 
stiffness, damping stiffness and bending stiffness. I plan to collect 10 values for each parameters so I will get 1000 
groups of physics parameter settings of cloth. In each group of setting, I collect data each 50 frame in the simulation 
with different camera position. The result is saved as png file and npz file.

##### Environment setting
+ Objects setting:
    - Table (with collision, is fixed at its base position and orientation)
    - Cloth (soft body, with collision, movable)
+ Physics setting:
    - Gravity: [0, 0, -10.0] at the all 1500 frames of simulation
    - initial velocity [0, 0, 0.5] for cloth at the first 500 frames of simulation

##### Physics parameters range
+ Elastic Stiffness range: [40.0, 130.0], 10.0 in each step, 10 parameters in total
+ Damping Stiffness range: [0.1, 1.0], 0.1 in each step, 10 parameters in total
+ Bending Stiffness range: [2.0, 20.0], 2.0 in each step, 10 parameters in total

###### Dataset
+ The number of data: 150,000 image in total (1000 physics parameter groups, 30 times of records for one group, 5 camera 
position for each time of record)
+ Dataset directory: ```path to project/simulation/data```  
    - Label: In the filename, ```(1)_(2)_(3)_(4)_(5)_(6).png or (1)_(2)_(3)_(4)_(5)_(6).npz```
        + (1) value of springElasticStiffness
        + (2) value of springDampingStiffness
        + (3) value of springBendingStiffness
        + (4) Iteration of data collection
        + (5) Camera position, 5 kind of position in total, (labeled as 0, 1, 2, 3, 4)
    - RGB directory: ```path to project/simulation/data/rgb```  
        + PNG file
    - Data directory: ```path to project/simulation/data/bin```
        + NPZ file, an compressive file for binary numpy array including raw depth data (include depth of table and cloth), 
        segmentation data (include segmentation of table and cloth) and depth data (only include cloth) 
        + How to open: 
        ```python
      import numpy as np
      import matplotlib.pyplot as plt
      
      data = np.load('path to file')
      raw_depth = data['raw_depth']
      segmentation = data['segmentation']
      depth = data['depth']
      
      plt.imshow(depth) # or plt.imshow(raw_depth) or plt.imshow(segmentation)
      plt.show()
        ```
 
#### How to use the code of simulation
1. Install all necessary packages by using ```pip install -r requirements.txt``` 
2. Run the script directly ```python ./simulation/init_world.py```

