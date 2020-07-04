# garment-detection
## English
### How to run

## 中文

### 如何运行simulation
+ ```python simulation/init_world.py``` 

### 如何查看结果
npy文件请用numpy读取，matplotlib输出，或者运行```python simulation/test.py```

### 运行结果
+ 结果目录 ```./simulation/data```
+ RGB数据 ```./simulation/data/rgb```  
    - PNG格式图片
+ 深度数据 ```./simulation/data/depth```  
    - npy文件(二进制文件，内部是二维数组)，包含环境内所有物体
+ 最终深度数据 ```./simulation/data/final_depth```
    - npy文件(二进制文件，内部是二维数组)，只含cloth
+ 边缘数据 ```./simulation/data/segmentation```
    - npy文件(二进制文件，内部是二维数组)，包含环境内所有物体

### 
