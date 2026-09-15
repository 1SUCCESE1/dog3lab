# dog3lab

dog3 四足机器人（12-DOF，纯腿式）的强化学习训练任务，按 [InstinctLab](../InstinctLab-main) 的风格接入，
独立成库以便单独演进（不污染原 InstinctLab 的 D1/D1H 工作）。

- 仿真后端：Isaac Sim 5.1 + Isaac Lab
- 算法：Instinct-RL（PPO），复用 `instinctlab` 的 env entry_point 与 wrappers
- gym id：`Instinct-Locomotion-Flat-Dog3-v0`、`Instinct-Locomotion-Rough-Dog3-v0`（各带 `-Play-v0`）
- 配套仿真部署仓库：[dog3_sim2sim](https://github.com/1SUCCESE1/dog3_sim2sim)（Gazebo 里跑同一份策略）

## 训练效果

Flat 地形，4096 envs × 6000 iterations：

| 指标 | 值 |
|---|---|
| 线速度跟踪奖励 `track_lin_vel_xy_exp` | **1.305** / 1.5 |
| 角速度跟踪 `track_ang_vel_z_exp` | 0.625 / 0.7 |
| 速度跟踪误差 `error_vel_xy` | **0.325 m/s** |
| Episode 长度 | **994** / 1000 |
| 摔倒率（`illegal_contact` 终止） | **0.99%** |
| 零命令站姿左右不对称 | **< 0.04 rad**（≈2°，近对称） |
| 步态 | **对角 trot，1.00 Hz，四腿同频** |

## 目录结构

```
dog3lab/
├── run.sh                                  # 统一入口：train / play / plot / python
├── scripts/instinct_rl/
│   ├── train.py  play.py  cli_args.py      # 训练 / 回放导出 / 参数（import dog3lab.tasks）
│   ├── smoke_dog3.py                       # headless 冒烟测试（建环境+步进，不训练）
│   ├── record_gait.py                      # 采关节轨迹，用于步态分析
│   └── keyboard_vel_controller.py          # play --keyboard 的终端键盘控制器
└── source/dog3lab/dog3lab/
    ├── assets/dog3.py                      # DOG3_CFG（URDF 资产 / 执行器 / 初始状态）
    ├── assets/resources/dog3_description/  # urdf + meshes
    └── tasks/dog3/
        ├── __init__.py                     # gym.register
        ├── config/{dog3_env_cfg,dog3_flat_env_cfg,agents/}
        └── mdp/{rewards,observations,curriculums}
```

## 安装

```bash
/home/you/isaacsim_5.1/python.sh -m pip install -e source/dog3lab --no-deps --no-build-isolation
```

`instinctlab` 需保持已安装（env entry_point 与 wrappers 复用它）。

## 训练 / 回放 / 导出

```bash
# 训练
./run.sh train --task=Instinct-Locomotion-Flat-Dog3-v0 --headless --num_envs=4096 --max_iterations=6000

# 回放（图形界面，键盘可驾驶：W/S 前后、A/D 侧移、Q/E 转向、空格停）
./run.sh play --task=Instinct-Locomotion-Flat-Dog3-Play-v0 \
              --load_run=<run_id> --checkpoint=model_6000.pt --num_envs=1 --keyboard

# 录视频（headless，产物 mp4 在 run 目录的 videos/play/）
./run.sh play --task=Instinct-Locomotion-Flat-Dog3-Play-v0 --headless --num_envs=1 \
              --load_run=<run_id> --checkpoint=model_6000.pt --video --video_length=600

# 导出 ONNX（给 dog3_sim2sim 部署用）
./run.sh play --task=Instinct-Locomotion-Flat-Dog3-Play-v0 --headless --num_envs=1 \
              --load_run=<run_id> --checkpoint=model_6000.pt --exportonnx
# 产物：logs/instinct_rl/dog3_locomotion_flat/<run>/exported/actor.onnx

# 冒烟 / 采步态
./run.sh python scripts/instinct_rl/smoke_dog3.py --headless --num_envs=16
./run.sh python scripts/instinct_rl/record_gait.py --task=Instinct-Locomotion-Flat-Dog3-Play-v0 \
    --headless --load_run=<run_id> --checkpoint=model_6000.pt --cmd_vx=0.5 --steps=400 --out=/tmp/gait.txt
```

## 机器人 / 任务参数

| 项 | 值 |
|---|---|
| DOF | 12（FL/FR/RL/RR × hip/thigh/calf） |
| 关节顺序 | FL, FR, RL, RR × hip, thigh, calf（**观测与动作同序**） |
| 关节力矩 / 速度上限 | 20 N·m；hip 30、thigh·calf 17.8 rad/s |
| base_link 质量 | 5.6022 kg |
| 执行器 | `DelayedPDActuator`，**kp 40 / kd 1.0** / armature 0.05 / friction 0.1 |
| 动作 | 12 维位置目标，hip scale 0.125、thigh/calf 0.25，`use_default_offset=True` |
| actor 观测 | **51 维**：ang_vel×0.25 / gravity / cmd×[2,2,0.25] / dof_pos / dof_vel×0.05 / last_action / **gait_phase(6)** |
| 控制频率 | 物理 dt 0.005 × decimation 4 → **50 Hz** |

### 关节零位约定

URDF 的关节 origin 已**整体平移**，使**关节零位 = 默认站姿**（原本 thigh +0.7 rad、calf −1.44 rad
的偏移写进 origin 的 rpy），关节限位同步平移。因此站立时所有关节角读到 **0**，
`DOG3_CFG.init_state.joint_pos` 全为 0。

这样做是为了让训练端与仿真部署端（`dog3_sim2sim`）使用**完全一致的模型与约定**，
避免两边角度基准不同导致策略迁移时姿态出错。

## 步态设计

奖励分为四组（见 `config/dog3_env_cfg.py`）：

- **跟踪**：`track_lin_vel_xy_exp` (+1.5)、`track_ang_vel_z_exp` (+0.7)
- **姿态**：`base_height_l2` (−10，目标 0.30 m)、`flat_orientation_l2` (−2)、`lin_vel_z_l2` (−2)、
  `ang_vel_xy_l2` (−0.05)
- **正则**：`action_rate_l2` (−0.03)、`joint_acc_l2` (−1e-6)、`joint_power` (−2e-5)、
  `joint_pos_limits` (−5)、`joint_vel_limits` (−1)、`default_joint_l2` (−0.02)
- **接触 / 步态**：
  - `stand_posture` (**−2.0**)：**零命令时**把关节拉回默认站姿 —— 站立对称的关键
  - `feet_gait` (**+0.3**)：按虚拟相位奖励"该摆动的对角腿确实离地"（trot 塑形）
  - `feet_air_time` (**+1.0**，阈值 0.35 s)：奖励更长的摆动相 → 更大步幅
  - `feet_slide` (−0.12)、`feet_stumble` (−0.1)、`undesired_contacts` (−1)、`contact_forces` (−1e-4)

**步态相位**：`gait_phase` 观测输出 6 维 `[sin φ, cos φ, sin(φ/2), cos(φ/2), sin(φ/4), cos(φ/4)]`，
`φ = 2πt/T`，`T = GAIT_CYCLE_TIME = 1.0 s`。这个布局与部署端 `rl_controller` 的 `phases`
观测**逐位对应**，因此策略不需要任何 C++ 改动就能部署。

## 当前状态

- Flat 地形已训练完成，指标见上表
- 已导出 ONNX 并在 `dog3_sim2sim` 的 Gazebo 仿真中部署验证
- Rough 地形任务（`Instinct-Locomotion-Rough-Dog3-v0`）已注册，可直接训练
