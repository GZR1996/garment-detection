# garment-detection
## English
### Simulation
#### Data collection plan

##### Environment setting

##### Physics parameters range
+ Elastic Stiffness range: [40.0, 130.0], 10.0 in each step
+ Damping Stiffness range: [0.1, 1.0], 0.1 in each step
+ Bending Stiffness range: [2.0, 20.0], 2.0 in each step

###### l


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

