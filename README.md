# mujocoquad_gym

## prerequisite

- conda (strongly recommended)
- [gym](https://gym.openai.com/) with every environment 
- [mujoco 1.5](http://www.mujoco.org/)

1. put mjpro150 directory into ```~/.mujoco```
2. put mjkey.txt into ```~/.mujoco```
3. install apt dependencies
    - see [gym README.md](https://github.com/openai/gym#installing-everything)
4. conda create -n mujocoquad python=3.6
5. conda activate mujocoquad 
6. export LD_LIBRARY_PATH
    ```sh
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
    $ # check your nvidia driver version 
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-410 
    ```
7. install gym by ```pip3 install 'gym[all]'```
8. install mujocoquad_gym by ```pip3 install -e .```

## how to run 

1. ```conda activate mujocoquad```
2. from the project root directory, 
    ```sh
    $ python quad_pid.py
    ```