# AUV 强化学习训练框架
## 1. 项目概述

本项目实现了一个用于水下航行器速度控制任务的强化学习训练环境。通过深度强化学习算法，训练 AUV 能够跟踪给定的目标速度指令。

```
auv_mjlab/
├── src/
│   ├── tasks/
│   │   └── velocity/              # Velocity 速度控制任务
│   │       ├── config/
│   │       │   └── auv/
│   │       │       ├── env_cfgs.py    # 环境配置
│   │       │       └── rl_cfg.py       # RL 算法配置
│   │       ├── mdp/
│   │       │   ├── observations.py    # 观测函数
│   │       │   ├── rewards.py         # 奖励函数
│   │       │   ├── terminations.py     # 终止条件
│   │       │   └── velocity_command.py # 速度指令生成器
│   │       └── rl/
│   │           └── runner.py          # 训练运行器
│   └── assets/
│       └── robots/                     # 机器人模型配置
├── scripts/
│   ├── train.py                        # 训练脚本
│   ├── play.py                         # 回放脚本
│   └── list_envs.py                    # 环境列表
└── README.md
```

---

## 2. Velocity 速度控制讲解

`task/velocity` 是本项目的核心任务模块，专注于训练 AUV 的基于速度的运动控制算法。

### 2.1 任务目标

AUV 需要根据随机生成的速度指令，调整推进器和舵机，使实际速度能够快速、准确地跟踪目标速度。

### 2.2 环境配置

任务配置文件位于 `src/tasks/velocity/config/auv/env_cfgs.py`，主要包含以下组件：

####  场景配置（Scene）

- **环境数量**: 支持大规模并行训练
- **地形类型**: 平地（plane）水下环境
- **机器人**: CQU AUV 无人水下航行器

#### 动作空间（Actions）

AUV 具有两类执行器：

| 执行器类型 | 关节名称 | 作用 | 动作尺度 |
|-----------|---------|------|---------|
| 舵机 | `joint_servo_up`, `joint_servo_left` | 控制方向舵角度 | ±1.5 rad |
| 推进器 | `joint_prop_up`, `joint_prop_down`, `joint_prop_left`, `joint_prop_right` | 提供推进力 | ±200 N |

#### 指令空间（Commands）

速度指令生成器使用 `UniformVelocityCommandCfg`，生成的指令包括：

- **线速度指令**:
  - X轴（前后）: -1.0 ~ 2.0 m/s
  - Y轴（侧向）: -1.0 ~ 1.0 m/s
- **角速度指令**:
  - Z轴（偏航）: -1.0 ~ 1.0 rad/s
- **指令重采样时间**: 3~8秒

#### 观测空间（Observations）

Agent 接收的观测信息包括：

| 观测项 | 描述 | 维度 |
|-------|------|------|
| `command` | 目标速度指令 | 4 |
| `base_lin_vel` | 机身线速度 | 3 |
| `base_ang_vel` | 机身角速度 | 3 |
| `projected_gravity` | 重力投影向量 | 3 |
| `joint_pos` | 舵机角度 | 2 |
| `joint_vel` | 关节速度 | 6 |
| `last_action` | 上一帧动作 | 8 |

#### 奖励函数（Rewards）

奖励设计分为核心目标奖励和惩罚项：

**核心奖励**:
- `track_linear_velocity`: 跟踪目标线速度（权重 1.0）
- `track_angular_velocity`: 跟踪目标角速度（权重 1.0）

**惩罚项**:
- `body_orientation_l2`: 惩罚非水平姿态，保持平稳（权重 -1.0）
- `body_ang_vel`: 惩罚 Roll/Pitch 轴角速度，防止晃动（权重 -0.05）
- `action_rate_l2`: 惩罚动作变化率，防止电机抖动（权重 -0.01）
- `action_l2`: 惩罚动作幅值，鼓励节能（权重 -0.005）
- `dof_vel_z`: 惩罚 Z 轴线速度，防止上下窜动（权重 -0.05）

#### 6. 事件系统（Events）

- **关节重置**: 训练开始时重置机器人关节状态
- **水流扰动**: 每隔 5-10 秒随机施加外部推力，增强鲁棒性

#### 7. 终止条件（Terminations）

- `time_out`: episode 时间超时（默认 20 秒）
- `fell_over`: 机身姿态超过 70° 倾斜

---

## 3. 训练指南

### 3.1 环境要求

- Python 3.10+
- CUDA 11.8+（GPU 训练）
- Mujoco 物理引擎

### 3.2 训练

```bash
python scripts/train.py Mjlab-Velocity-Flat-CQU-AUV --env.scene.num-envs=并行数量 --agent.max-iterations=回合数量
```

### 3.3 回放

```bash
python scripts/play.py Mjlab-Velocity-Flat-CQU-AUV \
    --checkpoint-file logs/rsl_rl/auv_velocity/时间戳/model_第几个回合.pt \
    --num-envs 1 \
    --viewer viser \
    --no-terminations True
```

回放后会返回一个 Viser 可视化网址，可在浏览器中查看训练效果。

---

## 4. 远程桌面训练指南

### 4.1 远程端操作

1. 通过 VSCode 连接远程 SSH：

```bash
ssh robot1@10.253.33.188
```

> 注意：需要先连接到同一局域网，连接时关闭 VPN，连接后可开启

2. 在远程终端启动 VNC 服务（建议在 base 环境中）：

```bash
# 停止现有 VNC 会话
vncserver -kill :1

# 启动新的 VNC 会话
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
```

### 4.2 本地操作

在本地打开远程桌面：

```bash
vncviewer 10.253.33.188:1
```

**VNC 密码**: 147258
。如果遇到如下错误，关闭代理进入
```
Wed Apr  1 18:35:26 2026
 DecodeManager: Detected 20 CPU core(s)
 DecodeManager: Creating 4 decoder thread(s)
 CConn:       Connected to host 10.253.33.188 port 5901
```

---

## 5. 参考资料

- [Unitree RL Mujoco 官方仓库](https://github.com/unitreerobotics/unitree_rl_mjlab)
