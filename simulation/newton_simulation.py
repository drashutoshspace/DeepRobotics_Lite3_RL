"""
Newton Physics Simulator for DeepRobotics Lite3 quadruped.

Supports two modes:
  --mode standalone   (default) Runs ONNX policy inference directly in Python.
                      Self-contained sim2sim validation — no C++ controller needed.
  --mode udp          Drop-in replacement for mujoco_simulation.py.
                      Communicates with the existing C++ controller via UDP.

Usage:
  pip install newton onnxruntime numpy
  python newton_simulation.py                          # standalone, flat ground
  python newton_simulation.py --scene stair            # standalone, stairs
  python newton_simulation.py --mode udp               # UDP mode (launch C++ controller separately)
  python newton_simulation.py --device cpu             # force CPU
  python newton_simulation.py --num-envs 4             # multiple parallel envs (standalone only)

Controls (standalone mode):
  W/S   - forward / backward
  A/D   - strafe left / right
  Q/E   - turn left / right
  R     - reset robot
  ESC   - quit
"""

import argparse
import os
import sys
import time
import socket
import struct
import threading
from pathlib import Path

import numpy as np

try:
    import warp as wp
    import newton
    from newton.solvers import SolverMuJoCo
except ImportError:
    print("Newton is not installed. Install with: pip install newton")
    print("Requires NVIDIA GPU with driver >= 545, or CPU fallback.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants matching the C++ controller
# ---------------------------------------------------------------------------
DT = 0.001  # 1 kHz simulation
DECIMATION = 12  # policy runs every 12 steps (~83 Hz)
RENDER_INTERVAL = 10  # render every 10 steps (100 Hz)

# Joint ordering (must match MJCF actuator order)
JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]
NUM_JOINTS = 12

# Policy constants (from lite3_test_policy_runner_onnx.cpp)
OBS_DIM = 45
ACT_DIM = 12
OMEGA_SCALE = 0.25
DOF_VEL_SCALE = 0.05

DEFAULT_JOINT_POS = np.array(
    [0.0, -0.8, 1.6] * 4, dtype=np.float32
)
ACTION_SCALE = np.array(
    [0.125, 0.25, 0.25] * 4, dtype=np.float32
)
KP = 30.0
KD = 1.0
MAX_CMD_VEL = np.array([0.8, 0.8, 0.8], dtype=np.float32)

# Initial pose for standing
INIT_JOINT_POS = np.array(
    [0.0, -0.8, 1.6] * 4, dtype=np.float32
)
# Standing height: with default joint angles, leg vertical extent is ~0.286m + 0.022m foot radius
INIT_HEIGHT = 0.32

# Standup / settling phase before policy takes over
STANDUP_KP = 100.0   # stiffer PD for standup (matches C++ standup_state)
STANDUP_KD = 2.5
WARMUP_STEPS = 2000  # 2 seconds at 1kHz — let robot settle before policy

# UDP ports (for UDP mode, matching mujoco_simulation.py)
UDP_LOCAL_PORT = 20001  # receive commands from C++ controller
UDP_CTRL_IP = "127.0.0.1"
UDP_CTRL_PORT = 30010  # send state to C++ controller


# ---------------------------------------------------------------------------
# Quaternion / rotation utilities
# ---------------------------------------------------------------------------
def quat_xyzw_to_rotmat(qx, qy, qz, qw):
    """Convert XYZW quaternion to 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def quat_xyzw_to_rpy(qx, qy, qz, qw):
    """Convert XYZW quaternion to roll-pitch-yaw (radians)."""
    # MuJoCo convention: WXYZ, Newton: XYZW
    w, x, y, z = qw, qx, qy, qz
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def projected_gravity_from_quat(qx, qy, qz, qw):
    """Compute gravity direction in body frame: R^T @ [0, 0, -1]."""
    R = quat_xyzw_to_rotmat(qx, qy, qz, qw)
    return R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# ONNX Policy Wrapper
# ---------------------------------------------------------------------------
class ONNXPolicy:
    """Loads and runs the pretrained ONNX locomotion policy."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        print(f"[Policy] Loaded ONNX model: {model_path}")

        # Warm-up
        dummy = np.ones((1, OBS_DIM), dtype=np.float32)
        self.session.run(["actions"], {"obs": dummy})
        print("[Policy] Warm-up inference OK")

    def reset(self):
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)

    def infer(self, base_omega, proj_gravity, cmd_vel, joint_pos, joint_vel):
        """
        Construct observation and run policy.
        Returns: target joint positions (12,) in robot order.
        """
        # Scale inputs (matching C++ code)
        omega_scaled = base_omega * OMEGA_SCALE
        cmd_vel_scaled = cmd_vel * MAX_CMD_VEL
        joint_pos_offset = joint_pos - DEFAULT_JOINT_POS
        joint_vel_scaled = joint_vel * DOF_VEL_SCALE

        # Build observation: [omega(3), gravity(3), cmd(3), pos(12), vel(12), last_act(12)]
        obs = np.concatenate([
            omega_scaled,
            proj_gravity,
            cmd_vel_scaled,
            joint_pos_offset,
            joint_vel_scaled,
            self.last_action,
        ]).astype(np.float32).reshape(1, OBS_DIM)

        # Run inference
        action_raw = self.session.run(["actions"], {"obs": obs})[0].flatten()
        self.last_action = action_raw.copy()

        # Map policy output to robot joint targets
        # robot_order == policy_order in this setup (identity permutation)
        target_pos = action_raw * ACTION_SCALE + DEFAULT_JOINT_POS
        return target_pos


# ---------------------------------------------------------------------------
# Keyboard input (standalone mode)
# ---------------------------------------------------------------------------
class KeyboardController:
    """Non-blocking keyboard input for velocity commands."""

    def __init__(self):
        self.cmd_vel = np.zeros(3, dtype=np.float32)  # [forward, side, turn]
        self.reset_flag = False
        self.quit_flag = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()

    def _input_loop(self):
        """Read keyboard in a background thread."""
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self.quit_flag:
                    ch = sys.stdin.read(1).lower()
                    with self._lock:
                        if ch == 'w':
                            self.cmd_vel[0] = min(self.cmd_vel[0] + 0.2, 1.0)
                        elif ch == 's':
                            self.cmd_vel[0] = max(self.cmd_vel[0] - 0.2, -1.0)
                        elif ch == 'a':
                            self.cmd_vel[1] = min(self.cmd_vel[1] + 0.2, 1.0)
                        elif ch == 'd':
                            self.cmd_vel[1] = max(self.cmd_vel[1] - 0.2, -1.0)
                        elif ch == 'q':
                            self.cmd_vel[2] = min(self.cmd_vel[2] + 0.2, 1.0)
                        elif ch == 'e':
                            self.cmd_vel[2] = max(self.cmd_vel[2] - 0.2, -1.0)
                        elif ch == 'r':
                            self.reset_flag = True
                        elif ch == ' ':
                            self.cmd_vel[:] = 0.0
                        elif ch == '\x1b':  # ESC
                            self.quit_flag = True
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            # Fallback: no terminal control available (e.g. piped input)
            pass

    def get_cmd_vel(self):
        with self._lock:
            return self.cmd_vel.copy()

    def should_reset(self):
        with self._lock:
            flag = self.reset_flag
            self.reset_flag = False
            return flag

    def should_quit(self):
        return self.quit_flag


# ---------------------------------------------------------------------------
# Newton Simulation
# ---------------------------------------------------------------------------
class NewtonSimulation:
    """Newton-based simulation for DeepRobotics Lite3."""

    def __init__(self, args):
        self.args = args
        self.device = args.device

        # Resolve MJCF path
        base_dir = Path(__file__).resolve().parent.parent
        if args.scene == "flat":
            xml_path = base_dir / "models" / "description" / "mjcf" / "Lite3_base.xml"
        elif args.scene == "stair":
            xml_path = base_dir / "models" / "description" / "mjcf" / "Lite3_stair.xml"
        else:
            xml_path = Path(args.scene)

        if not xml_path.exists():
            raise FileNotFoundError(f"MJCF not found: {xml_path}")
        print(f"[Newton] Loading MJCF: {xml_path}")

        # Build model
        wp.init()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        SolverMuJoCo.register_custom_attributes(builder)

        builder.add_mjcf(
            str(xml_path),
            enable_self_collisions=False,
        )

        # Configure PD gains for all actuated DOFs
        # Newton joint_q layout for floating base: [x,y,z, qx,qy,qz,qw, j0,j1,...j11]
        # The free joint has 7 q-coords and 6 qd-coords
        # Actuated DOFs start at index 7 in joint_q and index 6 in joint_qd
        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 0.0  # we apply torques manually
            builder.joint_target_kd[i] = 0.0

        # Set initial joint positions
        # Newton joint_q includes the free joint: [x, y, z, qx, qy, qz, qw, ...]
        # We need to set both the floating base and the actuated joints
        if builder.joint_q is not None and len(builder.joint_q) >= 7 + NUM_JOINTS:
            # Set base position
            builder.joint_q[0] = 0.0   # x
            builder.joint_q[1] = 0.0   # y
            builder.joint_q[2] = INIT_HEIGHT  # z
            builder.joint_q[3] = 0.0   # qx
            builder.joint_q[4] = 0.0   # qy
            builder.joint_q[5] = 0.0   # qz
            builder.joint_q[6] = 1.0   # qw
            # Set actuated joints
            for i in range(NUM_JOINTS):
                builder.joint_q[7 + i] = float(INIT_JOINT_POS[i])

        # Replicate for multi-env (standalone only)
        if args.num_envs > 1 and args.mode == "standalone":
            multi_builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
            SolverMuJoCo.register_custom_attributes(multi_builder)
            multi_builder.replicate(builder, args.num_envs, spacing=(2.0, 2.0, 0.0))
            self.model = multi_builder.finalize(device=self.device)
            self.num_envs = args.num_envs
        else:
            self.model = builder.finalize(device=self.device)
            self.num_envs = 1

        print(f"[Newton] Model finalized on {self.device}, envs={self.num_envs}")

        # Allocate simulation buffers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Forward kinematics for initial state
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create solver (MuJoCo backend for physics matching)
        self.solver = SolverMuJoCo(
            self.model,
            iterations=100,
            ls_iterations=50,
        )

        # Viewer — try GL first, then Viser (web-based), then headless
        self.viewer = None
        if not args.headless:
            # Try OpenGL viewer (requires pyglet >= 2.0)
            try:
                self.viewer = newton.viewer.ViewerGL(headless=False)
                self.viewer.set_model(self.model)
                print("[Newton] OpenGL viewer initialized")
            except Exception as e:
                print(f"[Newton] ViewerGL failed: {e}")
                # Try Viser (web-based viewer, opens in browser)
                try:
                    self.viewer = newton.viewer.ViewerViser()
                    self.viewer.set_model(self.model)
                    print("[Newton] Viser web viewer initialized (check browser)")
                except Exception as e2:
                    print(f"[Newton] ViewerViser also failed: {e2}, running headless")

        # Mode-specific setup
        if args.mode == "standalone":
            policy_path = str(base_dir / "models" / "pretrained" / "policy.onnx")
            self.policy = ONNXPolicy(policy_path)
            self.keyboard = KeyboardController()
        elif args.mode == "udp":
            self._setup_udp()

        # Torque buffer
        self.torques = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.last_print_time = 0.0

        # Cache total DOF count per env for torque application
        total_qd = self.model.joint_qd.shape[0]
        self.dof_per_env = total_qd // self.num_envs  # 6 (free) + 12 (actuated) = 18

    def _setup_udp(self):
        """Initialize UDP sockets for C++ controller communication."""
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind(("0.0.0.0", UDP_LOCAL_PORT))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctrl_addr = (UDP_CTRL_IP, UDP_CTRL_PORT)

        self.kp_cmd = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.kd_cmd = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.pos_cmd = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.vel_cmd = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.tau_ff = np.zeros(NUM_JOINTS, dtype=np.float32)

        self.udp_thread = threading.Thread(target=self._udp_receiver, daemon=True)
        self.udp_thread.start()
        print(f"[UDP] Listening on 0.0.0.0:{UDP_LOCAL_PORT}")

    def _udp_receiver(self):
        """Receive joint commands from C++ controller: 12f kp | 12f pos | 12f kd | 12f vel | 12f tau."""
        fmt = f"{NUM_JOINTS}f" * 5
        expected = struct.calcsize(fmt)
        while True:
            data, _ = self.recv_sock.recvfrom(expected)
            if len(data) < expected:
                continue
            unpacked = struct.unpack(fmt, data)
            n = NUM_JOINTS
            self.kp_cmd = np.array(unpacked[0:n], dtype=np.float32)
            self.pos_cmd = np.array(unpacked[n:2*n], dtype=np.float32)
            self.kd_cmd = np.array(unpacked[2*n:3*n], dtype=np.float32)
            self.vel_cmd = np.array(unpacked[3*n:4*n], dtype=np.float32)
            self.tau_ff = np.array(unpacked[4*n:5*n], dtype=np.float32)

    def _read_state(self):
        """Read joint positions, velocities, and base state from Newton."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()

        # Floating base: joint_q = [x,y,z, qx,qy,qz,qw, j0..j11]
        base_pos = joint_q[0:3]
        base_quat_xyzw = joint_q[3:7]  # Newton XYZW
        joint_pos = joint_q[7:7 + NUM_JOINTS].astype(np.float32)

        # joint_qd = [vx,vy,vz, wx,wy,wz, j0_dot..j11_dot]
        base_linvel = joint_qd[0:3]
        base_angvel = joint_qd[3:6].astype(np.float32)  # body angular velocity
        joint_vel = joint_qd[6:6 + NUM_JOINTS].astype(np.float32)

        # Derived quantities
        qx, qy, qz, qw = base_quat_xyzw
        rpy = quat_xyzw_to_rpy(qx, qy, qz, qw)
        proj_grav = projected_gravity_from_quat(qx, qy, qz, qw)

        return {
            "base_pos": base_pos,
            "base_quat_xyzw": base_quat_xyzw,
            "base_rpy": rpy,
            "base_linvel": base_linvel,
            "base_angvel": base_angvel,
            "proj_gravity": proj_grav,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def _compute_torques_standalone(self, state, step):
        """Compute joint torques using ONNX policy (standalone mode)."""
        q = state["joint_pos"]
        dq = state["joint_vel"]

        if step < WARMUP_STEPS:
            # Warmup phase: stiff PD control to default standing pose
            # Matches the C++ StandUpState behavior (KP=100, KD=2.5)
            self.torques = STANDUP_KP * (DEFAULT_JOINT_POS - q) + STANDUP_KD * (0.0 - dq)
            if step == WARMUP_STEPS - 1:
                print("[Newton] Warmup complete, switching to RL policy control")
                self.policy.reset()
            return

        # RL policy control
        if step % DECIMATION == 0:
            cmd_vel = self.keyboard.get_cmd_vel()
            target_pos = self.policy.infer(
                state["base_angvel"],
                state["proj_gravity"],
                cmd_vel,
                state["joint_pos"],
                state["joint_vel"],
            )
            self._target_pos = target_pos

        if not hasattr(self, "_target_pos"):
            self._target_pos = DEFAULT_JOINT_POS.copy()

        # PD control: tau = kp * (q_target - q) + kd * (0 - dq)
        self.torques = KP * (self._target_pos - q) + KD * (0.0 - dq)

    def _compute_torques_udp(self, state):
        """Compute joint torques from UDP commands (UDP mode)."""
        q = state["joint_pos"]
        dq = state["joint_vel"]
        self.torques = (
            self.kp_cmd * (self.pos_cmd - q) +
            self.kd_cmd * (self.vel_cmd - dq) +
            self.tau_ff
        )

    def _send_state_udp(self, state, timestamp):
        """Send robot state to C++ controller via UDP."""
        rpy = state["base_rpy"]
        angvel = state["base_angvel"]
        # Approximate body acceleration (use gravity as placeholder since
        # Newton doesn't expose accelerometer sensor directly)
        R = quat_xyzw_to_rotmat(*state["base_quat_xyzw"])
        body_acc = R.T @ np.array([0.0, 0.0, 9.81], dtype=np.float32)

        payload = np.concatenate([
            np.array([timestamp], dtype=np.float64),
            rpy.astype(np.float32),
            body_acc.astype(np.float32),
            angvel.astype(np.float32),
            state["joint_pos"].astype(np.float32),
            state["joint_vel"].astype(np.float32),
            self.torques.astype(np.float32),
        ])
        fmt = "1d" + f"{len(payload) - 1}f"
        try:
            self.send_sock.sendto(
                struct.pack(fmt, *payload), self.ctrl_addr
            )
        except socket.error as e:
            print(f"[UDP send] {e}")

    def _apply_torques(self):
        """Write computed torques into Newton's control buffer."""
        # Build full joint force array (free joint DOFs + actuated DOFs)
        # Free joint has 6 qd-DOFs, followed by 12 actuated DOFs per env
        full_torques = np.zeros(self.dof_per_env * self.num_envs, dtype=np.float32)
        for env_idx in range(self.num_envs):
            offset = env_idx * self.dof_per_env + 6  # skip free joint DOFs
            full_torques[offset:offset + NUM_JOINTS] = self.torques
        self.control.joint_f.assign(full_torques)

    def _print_debug(self, state, step):
        """Print debug info at 0.5 Hz."""
        now = time.perf_counter()
        if now - self.last_print_time < 2.0:
            return
        self.last_print_time = now

        def fmt(arr):
            return "[" + ", ".join(f"{x:6.2f}" for x in arr) + "]"

        phase = "WARMUP" if step < WARMUP_STEPS else "RL POLICY"
        print(f"=== [Newton Step {step} | {phase}] ===")
        print(f"  RPY:       {fmt(state['base_rpy'])}")
        print(f"  Omega:     {fmt(state['base_angvel'])}")
        print(f"  Joint Pos: {fmt(state['joint_pos'])}")
        print(f"  Joint Vel: {fmt(state['joint_vel'])}")
        print(f"  Torque:    {fmt(self.torques)}")
        if self.args.mode == "standalone":
            cmd = self.keyboard.get_cmd_vel()
            print(f"  Cmd Vel:   {fmt(cmd)}")
        print(f"  Base Pos:  {fmt(state['base_pos'])}")

    def _reset_robot(self):
        """Reset robot to initial standing pose."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()

        # Reset base
        joint_q[0:3] = [0.0, 0.0, INIT_HEIGHT]
        joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quat XYZW
        joint_q[7:7 + NUM_JOINTS] = INIT_JOINT_POS

        # Zero velocities
        joint_qd[:] = 0.0

        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        if self.args.mode == "standalone":
            self.policy.reset()
            self._target_pos = DEFAULT_JOINT_POS.copy()
            self.keyboard.cmd_vel[:] = 0.0
            self._reset_step = True  # signal to reset step counter

        self.torques[:] = 0.0
        print("[Newton] Robot reset — warmup phase restarting")

    def start(self):
        """Main simulation loop."""
        print(f"[Newton] Starting simulation (mode={self.args.mode}, dt={DT})")
        if self.args.mode == "standalone":
            print(f"[Newton] Warmup phase: {WARMUP_STEPS} steps ({WARMUP_STEPS*DT:.1f}s) "
                  f"with KP={STANDUP_KP}, KD={STANDUP_KD}")
            print("[Controls] W/S=fwd/back, A/D=strafe, Q/E=turn, SPACE=stop, R=reset, ESC=quit")

        step = 0
        sim_time = 0.0
        last_wall_time = time.perf_counter()

        try:
            while True:
                # Real-time pacing
                now = time.perf_counter()
                if now - last_wall_time < DT:
                    continue
                last_wall_time = now

                # Check quit/reset (standalone)
                if self.args.mode == "standalone":
                    if self.keyboard.should_quit():
                        break
                    if self.keyboard.should_reset():
                        self._reset_robot()
                        step = 0  # restart warmup
                    if getattr(self, "_reset_step", False):
                        step = 0
                        self._reset_step = False

                # Check viewer
                if self.viewer and not self.viewer.is_running():
                    break

                # Read current state
                state = self._read_state()

                # Safety check (matching C++ PostureUnsafeCheck)
                rpy = state["base_rpy"]
                if abs(rpy[0]) > np.radians(60) or abs(rpy[1]) > np.radians(60):
                    print(f"[SAFETY] Posture unsafe: roll={np.degrees(rpy[0]):.1f}, "
                          f"pitch={np.degrees(rpy[1]):.1f} — resetting")
                    self._reset_robot()
                    state = self._read_state()

                # Compute torques
                if self.args.mode == "standalone":
                    self._compute_torques_standalone(state, step)
                else:
                    self._compute_torques_udp(state)
                    self._send_state_udp(state, sim_time)

                # Apply torques and step physics
                self._apply_torques()
                self.state_0.clear_forces()
                self.model.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
                self.state_0, self.state_1 = self.state_1, self.state_0

                step += 1
                sim_time = step * DT

                # Render
                if self.viewer and step % RENDER_INTERVAL == 0:
                    self.viewer.begin_frame(sim_time)
                    self.viewer.log_state(self.state_0)
                    self.viewer.end_frame()

                # Debug print
                self._print_debug(state, step)

        except KeyboardInterrupt:
            print("\n[Newton] Interrupted")
        finally:
            if self.viewer:
                self.viewer.close()
            print(f"[Newton] Simulation ended at step {step} (t={sim_time:.3f}s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Newton Physics Simulation for DeepRobotics Lite3"
    )
    parser.add_argument(
        "--mode", choices=["standalone", "udp"], default="standalone",
        help="standalone: self-contained with ONNX policy; udp: C++ controller via UDP"
    )
    parser.add_argument(
        "--scene", default="flat",
        help="Scene: 'flat' (Lite3_base.xml), 'stair' (Lite3_stair.xml), or path to MJCF"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Compute device: 'cuda:0', 'cpu', etc."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without viewer"
    )
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Number of parallel environments (standalone mode only)"
    )
    args = parser.parse_args()

    sim = NewtonSimulation(args)
    sim.start()


if __name__ == "__main__":
    main()
