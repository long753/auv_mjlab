"""CQU AUV velocity environment configurations."""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
import src.tasks.velocity.mdp as mdp
from src.assets.robots import get_auv_robot_cfg


def cqu_auv_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:

    # ==========================
    # 1. 场景配置 (Scene)
    # ==========================
    scene = SceneCfg(
        num_envs=1, # 默认环境数，RL 运行时会被覆盖
        extent=2.0,
        entities={"robot": get_auv_robot_cfg()},
        terrain=TerrainEntityCfg(terrain_type="plane"), # 使用平地
        sensors=(), # 移除了射线扫描等不必要的传感器
    )

    # ==========================
    # 2. 动作空间 (Actions)
    # ==========================
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=('joint_servo_up',        # 舵机角度（竖轴）
                            'joint_servo_left'),      # 舵机角度（横轴）
            scale=1.5,                               # 舵机角度范围
            use_default_offset=True,
        ),

        "thrusters": JointPositionActionCfg( 
            entity_name="robot",
            actuator_names=('joint_prop_up',         # 上侧推进器推力
                            'joint_prop_down',       # 上侧推进器推力
                            'joint_prop_left',       # 上侧推进器推力
                            'joint_prop_right'),     # 上侧推进器推力
            scale=200,                                # 推进器推力范围
            use_default_offset=False,
        )
    }

    # ==========================
    # 3. 目标指令生成器 (Commands) 
    # ==========================
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.05,
            rel_heading_envs=0.25,
            heading_command=True,
            heading_control_stiffness=0.5,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.0), # 前后速度范围
                lin_vel_y=(-1.0, 1.0), # 侧向速度范围
                ang_vel_z=(-1.0, 1.0), # 偏航角速度范围
                heading=(-math.pi, math.pi),
            ),
        )
    }

    # ==========================
    # 4. 观测空间 (Observations)
    # ==========================
    actor_terms = {
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),   # 目标指令
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),                                      # 机身线速度
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),                                      # 机身角速度
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),                            # 重量投影
        "actions": ObservationTermCfg(func=mdp.last_action),                                            # 上一帧action
        
        "joint_pos": ObservationTermCfg(                 # 舵机转角
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(
                'joint_servo_up','joint_servo_left'
            ))},
        ),
        "joint_vel": ObservationTermCfg(                # 舵机和推进器速度
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(
                'joint_servo_up','joint_servo_left','joint_prop_up','joint_prop_down','joint_prop_left','joint_prop_right'
            ))},
        ),
    }
    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play, # 演示模式关闭噪声
            history_length=1,
        ),
        "critic": ObservationGroupCfg( # Critic 通常和 Actor 观测相同，或者更丰富
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False, # Critic 训练时不加噪声
            history_length=1,
        ),
    }

    # ==========================
    # 5. 奖励函数 (Rewards)
    # ==========================
    rewards = {
        # --- 核心目标奖励 ---
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        # --- 惩罚项 ---
        "body_orientation_l2": RewardTermCfg(
            func=mdp.body_orientation_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",))},
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",))},
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2, weight=-0.01
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2, 
            weight=-0.01
        ),
        "action_l2": RewardTermCfg(
            func=mdp.action_l2, 
            weight=-0.005
        ),
        "dof_vel_z": RewardTermCfg(
            func=mdp.base_lin_vel_penalty, 
            weight=-0.05,
            params={"axis": 2} 
        ),
    }

    # ==========================
    # 6. 事件 (Events) 
    # ==========================
    events = {
        # 1. 原有的关节重置
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_servo_(up|left)", "joint_prop_.*")),
            },
        ),
        # 2. 模拟水流扰动：每隔一定步数随机推一下机身
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0), # 每 5-10 秒推一次
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2)}},
        ),
    }

    # ==========================
    # 7. 终止条件 (Terminations)
    # ==========================
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),  # 超出一定时间
        "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),   # 与给定目标偏差过大
    }

    # ==========================
    # 8. 组装并返回最终配置
    # ==========================
    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        
        # --- 底层控制参数 ---
        metrics={},       # 不需要额外指标时给空字典
        curriculum={},    # 移除原有的地形课程学习
        decimation=4,     # 控制频率降采样 (关键参数，决定了控制频率！)
        episode_length_s=1e9 if play else 20.0,
        
        # --- 物理引擎与渲染 ---
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base", 
            distance=2.5, 
            elevation=-15.0
        ),
        sim=SimulationCfg(
            nconmax=1024,
            njmax=2048,
            contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
    )