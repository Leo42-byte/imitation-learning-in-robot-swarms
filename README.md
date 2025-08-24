# imitation-learning-in-robot-swarms

This repository contains the source code, ARGoS configuration files, and experimental results used in the dissertation **"Robot-to-Robot Imitation Learning in Robot Swarms"**.  
It implements and evaluates imitation learning experiments with single-teacher and multi-teacher scenarios using foot-bot robots in the ARGoS3 simulator.


## Installation

1. Install [ARGoS3] and ensure it runs correctly.  

2. Running Experiments：
argos3 -c imitation_learning.argos

Note: The demonstrator's trajectory shape and number of repetitions can be adjusted by modifying the variables polygon_sides and demo_repeat_target in the Lua controller file.
