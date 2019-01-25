#!/usr/bin/env python3
"""
controller model is from 

Seungho Jeong and Seul Jung, Position Control of a Quad-Rotor System, 2013
https://link.springer.com/chapter/10.1007/978-3-642-37374-9_94
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import numpy as np
import os

# world constants
dt = 0.001
gravity = 9.81

MODEL_XML = """
<mujoco model="quadrotor0">
	<compiler inertiafromgeom="true" coordinate="local"/>
	<option	timestep="{timestep}" gravity="0 0 -{gravity}" density="1" viscosity="1e-5" />
	<worldbody>
		<geom name="floor" pos="0 0 0" size="2 2 .2" type="plane"  conaffinity="1" rgba="1 1 1 1" condim="3"/>
		<body name="quadrotor" pos="0 0 0" >
			<geom name="core" type="box" pos="0 0 0" quat = "1. 0. 0. 0" size="0.06 0.035 0.025"  rgba="0.3 0.3 0.8 1" mass = ".1"/>
			
			<geom name="a00" type="box" pos=".071 0.071 0.0" size="0.05 0.01 0.0025"  quat = ".924 0.0 0.0 0.383" rgba="0.3 0.3 0.8 1" mass = ".025"/>
			<geom name="a10" type="box" pos=".071 -0.071 0.0" size="0.05 0.01 0.0025"  quat = ".383 0.0 0.0 0.924" rgba="0.3 0.3 0.8 1" mass = ".025"/>
			<geom name="a20" type="box" pos="-0.071 -0.071 0.0" size="0.05 0.01 0.0025"  quat = "-.383 0.0 0.0 0.924" rgba="0.3 0.3 0.8 1" mass = ".025"/>
			<geom name="a30" type="box" pos="-.071 0.071 0.0" size="0.05 0.01 0.0025"  quat = ".924 0.0 0.0 -0.383" rgba="0.3 0.3 0.8 1" mass = ".025"/>
			
			<joint name="root"   type="free" damping="0" armature="0" pos="0 0 0" />
			
			<!-- Motor sites to attach motor actuators --->
            <site name="motor0" type="cylinder" pos=" 0.1  0.1 0.01"  size="0.01 0.0025"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
            <site name="motor1" type="cylinder" pos=" 0.1 -0.1 0.01"  size="0.01 0.0025"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
            <site name="motor2" type="cylinder" pos="-0.1 -0.1 0.01"  size="0.01 0.0025"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
            <site name="motor3" type="cylinder" pos="-0.1  0.1 0.01"  size="0.01 0.0025"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
			
			<!-- Thruster geometries for collisions since site's are excluded from collision checking --->
            <geom name="thruster0" type="cylinder" pos=" 0.1  0.1  0.01" size="0.05 0.0025"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1" mass = ".025"/>
            <geom name="thruster1" type="cylinder" pos=" 0.1 -0.1  0.01" size="0.05 0.0025"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1" mass = ".025"/>
            <geom name="thruster2" type="cylinder" pos="-0.1 -0.1  0.01" size="0.05 0.0025"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1" mass = ".025"/>
            <geom name="thruster3" type="cylinder" pos="-0.1  0.1  0.01" size="0.05 0.0025"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1" mass = ".025"/>
            
            <!-- Visualization of the coordinate frame --->
			<site name="qcX" type="box" pos="0.1 0.0 0.0" size="0.1 0.005 0.005"  quat = " 1.000  0.0  0.0    0."     rgba="1 0 0 1" />
			<site name="qcY" type="box" pos="0.0 0.1 0.0" size="0.1 0.005 0.005"  quat = " 0.707  0.0  0.0    0.707"  rgba="0 1 0 1" />
			<site name="qcZ" type="box" pos="0.0 0.0 0.1" size="0.1 0.005 0.005"  quat = "-0.707  0.0  0.707  0."     rgba="0 0 1 1" />
		</body>
	</worldbody>
    <actuator>
        <motor ctrllimited="true" ctrlrange="0.0 1.0" gear="0  0. 1. 0. 0. 0." site="motor0"/>
        <motor ctrllimited="true" ctrlrange="0.0 1.0" gear="0  0. 1. 0. 0. 0." site="motor1"/>
        <motor ctrllimited="true" ctrlrange="0.0 1.0" gear="0  0. 1. 0. 0. 0." site="motor2"/>
        <motor ctrllimited="true" ctrlrange="0.0 1.0" gear="0  0. 1. 0. 0. 0." site="motor3"/>
	</actuator>
</mujoco>
""".format(timestep=dt, gravity=gravity)

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)

t = 0
ex = 0
es = 0
ex_int = 0

#####################################
# constants
mass = model.body_mass[1]
L = 0.1     # moment arm (L_arm cos 45)


class Trajectory:
    R = 0.5     # trajectory radius
    w = 1.0     # trajectory angular speed (rad/s)

#####################################
# control matrix


class CtrlParam:
    #################################
    # attitude

    kpz = 2.
    kpphi = 0.1
    kptheta = 0.1
    kppsi = 0.1

    Kx_p = np.array([
        [kpz, 0, 0, 0],
        [0, kpphi, 0, 0],
        [0, 0, kptheta, 0],
        [0, 0, 0, kppsi],
    ])

    kdz = 0.5
    kdphi = 0.1
    kdtheta = 0.1
    kdpsi = 0.1

    Kx_d = np.array([
        [kdz, 0, 0, 0],
        [0, kdphi, 0, 0],
        [0, 0, kdtheta, 0],
        [0, 0, 0, kdpsi],
    ])

    kiz = 0.01
    kiphi = 0.01
    kitheta = 0.01
    kipsi = 0.01

    Kx_i = np.array([
        [kiz, 0, 0, 0],
        [0, kiphi, 0, 0],
        [0, 0, kitheta, 0],
        [0, 0, 0, kipsi],
    ])

    #################################
    # position control matrix

    kpx = 0.6
    kpy = 0.6

    Ks_p = np.array([
        [kpx, 0],
        [0, kpy],
    ])

    kdx = 0.2
    kdy = 0.2

    Ks_d = np.array([
        [kdx, 0],
        [0, kdy],
    ])

#####################################
# rotor matrix

C = 0.1     # constant factor

a = 0.25
b = 1 / (4*L)
c = 1 / (4*C)

C_R = np.array([
    [a, b, -b, -c],
    [a, -b, -b, c],
    [a, -b, b, -c],
    [a, b, b, c],
])

#####################################
# loop
while True:

    #################################
    # desired position state (x, y, z)
    # circle trajectory on 1 m height
    s_d = np.array([
        Trajectory.R * np.cos(Trajectory.w * dt * t),
        Trajectory.R * np.sin(Trajectory.w * dt * t),
        1.0
    ])

    ################################
    # rpy
    rotmat_WB = sim.data.get_body_xmat('quadrotor')

    yaw = np.arctan2(rotmat_WB[1][0], rotmat_WB[0][0])
    pitch = np.arctan2(-rotmat_WB[2][0], np.sqrt(rotmat_WB[2][1] ** 2 + rotmat_WB[2][2] ** 2))
    roll = np.arctan2(rotmat_WB[2][1], rotmat_WB[2][2])

    ################################
    # state

    # position
    s = np.array([
        sim.data.get_body_xipos('quadrotor')[0],
        sim.data.get_body_xipos('quadrotor')[1],
    ])

    # attitude
    x = np.array([
        sim.data.get_body_xipos('quadrotor')[2],
        roll,
        pitch,
        yaw,
    ])

    ################################
    # error

    # position
    es_last = es
    es = s_d[0:2] - s
    es_dot = (es - es_last) / dt            # differentiation

    rotmat_BW = np.linalg.inv(rotmat_WB)

    # position input
    us = np.matmul(CtrlParam.Ks_p, es) \
         + np.matmul(CtrlParam.Ks_d, es_dot)
    us = np.append(us, 0)

    # attitude
    x_d = np.array([
        s_d[2],                             # +z
        -np.matmul(rotmat_BW, us)[1],       # -y -> roll,
        np.matmul(rotmat_BW, us)[0],        # +x -> pitch,
        0.0,
    ])

    ex_last = ex
    ex = x_d - x
    ex_dot = (ex - ex_last) / dt            # differentiation
    ex_int += ex * dt                       # integration

    # attitude input
    u = np.matmul(CtrlParam.Kx_p, ex) \
        + np.matmul(CtrlParam.Kx_d, ex_dot) \
        + np.matmul(CtrlParam.Kx_i, ex_int)
    u[0] += mass * gravity / (np.cos(pitch) * np.cos(roll))

    # actuator input
    F = np.matmul(C_R, u)

    sim.data.ctrl[0] = F[0]     # +,+
    sim.data.ctrl[1] = F[1]     # +,-
    sim.data.ctrl[2] = F[2]     # -,-
    sim.data.ctrl[3] = F[3]     # -,+

    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
