# chainsaw-robot
Simulating a robotic chainsaw carving a 3D model


## To Generate Graphs
```
python main_results.py
```

## To actually cut models and print out policy results
```
python main.py
```
In this file, I have marked the lines one can modify in order to change the cut choosing policy and the number of cuts the algorithm should make

## Requirements
- alphashape==1.3.1
- matplotlib==3.7.2
- numpy==1.25.2
- scipy==1.11.4
- trimesh==4.0.5
