# Lite3 RL Locomotion

Reinforcement learning locomotion controller for the [Deep Robotics Lite3](https://www.deeprobotics.cn/en/products/Lite3.html) quadruped. Supports sim-to-sim validation (MuJoCo/PyBullet) and sim-to-real deployment.

## Repository Structure

```
.
├── src/                    # C++ controller source
│   ├── main.cpp
│   ├── state_machine/      # FSM: Idle -> StandUp -> RL -> JointDamping
│   ├── interface/          # Robot (sim/hardware) and input (keyboard/gamepad)
│   └── *policy_runner*     # ONNX inference wrapper
├── simulation/             # Python simulation scripts
│   ├── mujoco_simulation.py
│   └── pybullet_simulation.py
├── models/
│   ├── pretrained/         # PPO policy (ONNX) + pt-to-onnx converter
│   └── description/        # MJCF files + meshes
├── deploy_scripts/         # SCP/SFTP scripts for robot deployment
├── third_party/            # Eigen, ONNX Runtime, MuJoCo, Gamepad SDK, Motion SDK
└── CMakeLists.txt
```

## Quick Start (Sim-to-Sim)

### Prerequisites

```bash
sudo apt-get install libdw-dev
pip install pybullet "numpy<2.0" mujoco colorama
```

### Build

```bash
mkdir build && cd build
cmake .. -DBUILD_PLATFORM=x86 -DBUILD_SIM=ON -DSEND_REMOTE=OFF
make -j
```

### Run

```bash
# Terminal 1: simulation
cd simulation/
python mujoco_simulation.py

# Terminal 2: RL controller
cd build/
./rl_deploy
```

### Controls (in Terminal 2)

**State transitions:**
- `z` - stand up (default position)
- `c` - switch to RL control mode

**Movement (in RL mode):**
- `w/s` - forward / backward
- `a/d` - strafe left / right
- `q/e` - rotate left / right

**Stopping:**
- `r` - emergency stop (enters joint damping mode, works from any state)
- `z` - return to standing pose (controlled transition out of RL mode)
- `Ctrl+C` - kill the process

## Policy

The pretrained policy is at `models/pretrained/policy.onnx` (PPO, 744 KB).

**Observation space (45-dim):** angular velocity (3), projected gravity (3), velocity command (3), joint positions (12), joint velocities (12), last action (12).

**Action space (12-dim):** target joint positions for all 4 legs (hip_x, hip_y, knee per leg).

Proprioceptive only, no vision or contact sensors. Trained on randomized terrain so it handles stairs, slopes, and rough ground through reactive gait adaptation.

To use a custom trained policy, convert `.pt` to `.onnx`:
```bash
cd models/pretrained/
python pt2onnx.py
```

## Sim-to-Real Deployment

1. Connect to robot WiFi (`Lite*******`, password: `12345678`)
2. SCP the repo to the robot: `scp -r . ysc@192.168.2.1:~/Lite3_rl`
3. SSH in: `ssh ysc@192.168.2.1`
4. Build for ARM:
```bash
mkdir build && cd build
cmake .. -DBUILD_PLATFORM=arm -DBUILD_SIM=OFF -DSEND_REMOTE=OFF
make -j
./rl_deploy
```

## Acknowledgements

Built on top of [Deep Robotics](https://github.com/DeepRoboticsLab) open-source resources (BSD-3 license).
