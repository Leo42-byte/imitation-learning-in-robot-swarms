# Robot-to-robot imitation learning in robot swarms
Author:Qi Liu

Msc Robotics, University of Bristol & University of the West of England

This repository contains the source code, ARGoS configuration files, and experimental results used in the dissertation **"Robot-to-Robot Imitation Learning in Robot Swarms"**.  
It implements and evaluates imitation learning experiments with single-teacher and multi-teacher scenarios using foot-bot robots in the ARGoS3 simulator.


## Installation and Run Experiments
This project has been tested with **Ubuntu 20.04 LTS** and **ARGoS3**.  
1. Install [ARGoS3] and ensure it runs correctly.  

2. Running Experiments：
argos3 -c imitation_learning.argos

## File Structure

### imitation_learning.argos
ARGoS configuration file. Defines the simulation environment (arena, robots, sensors, actuators, etc.) and links the Lua controller.

### controller.lua
Main controller script for the foot-bot robots. Implements the demonstrator and learner behaviors, including trajectory generation, observation, and imitation logic.

### result/
Contains outputs from the experiments:

exp1/: imitation quality scores Q for the single-teacher experiments.

exp2/: includes both the demonstrators’ trajectory variance scores S  and the learner’s imitation quality scores Q.

Note: The demonstrator's trajectory shape and number of repetitions can be adjusted by modifying the variables polygon_sides and demo_repeat_target in the Lua controller file.
