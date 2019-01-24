# mujocoquad

## prerequisite

- conda (strongly recommended)
- mujoco 1.5
- openai gym (optional)

1. put mjpro150 directory into ```~/.mujoco```
2. put mjkey.txt into ```~/.mujoco```
3. conda create -n mujocoquad python=3.6
4. conda activate mujocoquad 
4. export LD_LIBRARY_PATH
```.sh
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/donghok/.mujoco/mjpro150/bin
$ # check your nvidia driver version 
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-410 
```
5. install mujoco_py by ```pip3 install -U 'mujoco-py<1.50.2,>=1.50.1'```