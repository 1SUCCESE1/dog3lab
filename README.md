# dog3lab

dog3 四足机器人（12-DOF，纯腿式）的强化学习训练任务，按 [InstinctLab](../InstinctLab-main) 的风格接入，
独立成库以便单独演进（不污染原 InstinctLab 的 D1/D1H 工作）。

- 仿真后端：Isaac Sim 5.1 + Isaac Lab
- 算法：Instinct-RL（PPO），复用 `instinctlab` 的 env entry_point 与 wrappers
- gym id：`Instinct-Locomotion-Flat-Dog3-v0`、`Instinct-Locomotion-Rough-Dog3-v0`（各带 `-Play-v0`）

## 目录结构

```
dog3lab/
├── run.sh                                  # 统一入口：train / play / plot / python
├── scripts/instinct_rl/
│   ├── train.py  play.py  cli_args.py      # 训练 / 回放 / 参数（import dog3lab.tasks）
│   ├── smoke_dog3.py                       # headless 冒烟测试（建环境+步进，不训练）
│   └── record_gait.py                      # 采关节轨迹，用于步态分析
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
# 用 isaacsim 的 kit python 做 editable 安装
/home/you/isaacsim_5.1/python.sh -m pip install -e source/dog3lab --no-deps --no-build-isolation
```

`instinctlab` 需保持已安装（env entry_point 与 wrappers 复用它）。

## 训练 / 回放

```bash
./run.sh train --task=Instinct-Locomotion-Flat-Dog3-v0 --headless --num_envs=4096 --max_iterations=6000
./run.sh play  --task=Instinct-Locomotion-Flat-Dog3-Play-v0 --headless --num_envs=1 \
               --load_run=<run_id> --checkpoint=model_6000.pt --exportonnx
./run.sh python scripts/instinct_rl/smoke_dog3.py --headless --num_envs=16
```

## 机器人 / 任务关键参数

| 项 | 值 |
|---|---|
| DOF | 12（FL/FR/RL/RR × hip/thigh/calf） |
| 关节顺序 | FL, FR, RL, RR × hip, thigh, calf（动作与观测一致） |
| 关节力矩 / 速度上限 | 20 N·m；hip 30、thigh·calf 17.8 rad/s |
| base_link 质量 | 5.6022 kg |
| 默认站姿 | hip 0 / thigh 0 / calf 0（见下"关节零位约定"） |
| 动作 | 12 维位置目标，hip scale 0.125、thigh/calf 0.25，`use_default_offset=True` |
| actor 观测 | 45 维：ang_vel×0.25 / gravity / cmd×[2,2,0.25] / dof_pos / dof_vel×0.05 / last_action |
| 控制频率 | 物理 dt 0.005 × decimation 4 → 50 Hz |
| 执行器 | `DelayedPDActuator`，kp 20 / kd 0.5 / armature 0.05（占位值，待标定） |

### 关节零位约定

URDF 的关节 origin 已**整体平移**，使**关节零位 = 默认站姿**（thigh +0.7 rad、calf −1.44 rad 的偏移被
写进 origin 的 rpy），关节限位同步平移。因此：

- 站立时所有关节角读到 **0**
- `DOG3_CFG.init_state.joint_pos` 全为 0

这么做是为了让 Isaac 训练端与 sim2sim（Gazebo/MuJoCo，见 `dog3_sim2sim`）使用**完全一致的模型**，
避免两边约定不同导致策略迁移时步态退化。后足 link（`LH_FOOT`/`RH_FOOT`）相对小腿**后移 0.035 m**，
把后腿着地点后移以匹配重心（否则后腿会在自重下塌陷）。

## 已知状态

- Flat 训练到 6000 iter：episode 长度 1000/1000、摔倒率 ~0.8%、`error_vel_xy` ~0.29 m/s，
  学到的是**对角 trot（1.38 Hz）**
- 尚未做：Rough 地形、`gait_trot`/`foot_clearance` 步态塑形奖励、kp/kd/armature 标定
