# garment-detection
## English
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
+ The number of data: 120,000 image in total
+ 
+ Dataset directory: ```path to project/simulation/data```  
    - Label: In the filename, ```(1)_(2)_(3)_(4)_(5)_(6).png or (1)_(2)_(3)_(4)_(5)_(6).npz```
        + (1) value of springElasticStiffness
        + (2) value of springDampingStiffness
        + (3) value of springBendingStiffness
        + (4) Iteration of data collection
        + (5) Camera position, 5 kind of position in total, (labeled as 0, 1, 2, 3, 4)
    - RGB directory: ```path to project/simulation/data/rgb```  
        + PNG file
        + Example: []
    - Data directory: ```path to project/simulation/data/bin```
        + NPZ file, an compressive file including raw depth data, segmentation data and depth data  
        + Example: []
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



### Neural network


## 中文

### 仿真过程

#### 如何运行simulation
+ ```python simulation/init_world.py``` 

#### 如何查看结果
npy文件请用numpy读取，matplotlib输出，或者运行```python simulation/test.py```

###### 文件名含义
文件名例子: (1)_(2)_(3)_(4)_(5)_(6)_(7)
+ (1): springElasticStiffness: 弹性系数
+ (2): springDampingStiffness：阻尼系数
+ (3): springBendingStiffness: 弯曲系数
+ (4): pointsToHold: 抓取点的个数
+ (5): holdAnchorIndex: 抓取布的index节点
+ (6): iteration: simulation过程次数
+ (7): eyePosition: 摄像机位置

###### 运行结果
+ 结果目录 ```./simulation/data```
+ RGB数据 ```./simulation/data/rgb```  
    - PNG格式图片
+ 深度数据 ```./simulation/data/depth```  
    - npy文件(二进制文件，内部是二维数组)，包含环境内所有物体
+ 最终深度数据 ```./simulation/data/final_depth```
    - npy文件(二进制文件，内部是二维数组)，只含cloth
+ 边缘数据 ```./simulation/data/segmentation```
    - npy文件(二进制文件，内部是二维数组)，包含环境内所有物体

