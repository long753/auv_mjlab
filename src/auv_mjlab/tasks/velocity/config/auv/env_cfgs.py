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
import src.auv_mjlab.tasks.velocity.mdp as mdp
from src.auv_mjlab.assets.robots import get_auv_robot_cfg
from src.auv_mjlab.assets.robots.auv.auv_constants import AUV_MAX_SERVO_ANGLE, get_auv_physical_params

# 从模型中动态获取水动力参数
_phys = get_auv_physical_params()
AUV_MASS = _phys["mass"]
AUV_VOLUME = _phys["volume"]
AUV_INERTIA = list(_phys["inertia"])
AUV_COM_TO_COB_OFFSET = list(_phys["com_to_cob_offset"])


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
        "joint_pos": mdp.AuvThrusterAllocationActionCfg(
            entity_name="robot",
            servo_joint_names=["joint_servo_up", "joint_servo_left"],  # 舵机
            thruster_joint_names=[                                     # 推进器
                "thrust_up_site",
                "thrust_down_site",
                "thrust_left_site",
                "thrust_right_site",
            ],
            max_servo_angle=AUV_MAX_SERVO_ANGLE, 
            max_thrust=50.0,
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
                lin_vel_x=(-1.0, 2.0),
                lin_vel_y=(-1.0, 1.0), 
                ang_vel_z=(-1.0, 1.0), 
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
        
        # 仅观测主动控制的两个舵机转角
        "joint_pos": ObservationTermCfg(                 
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(
                'joint_servo_up','joint_servo_left'
            ))},
        ),
        # 观测所有活动关节的速度
        "joint_vel": ObservationTermCfg(                
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(
                'joint_servo_up','joint_servo_down','joint_servo_left','joint_servo_right'
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
        "align_z_with_velocity": RewardTermCfg(
            func=mdp.align_z_with_velocity,  # 函数引用
            weight=1.0,                      # 奖励权重
            params={
                "command_name": "twist",     # 命令名称（对应 velocity_command）
                "std": 0.5,                  # 指数奖励宽度参数
            },
        ),
    }

    # ==========================
    # 6. 事件 (Events) 
    # ==========================
    events = {
        # 根body重置（给初速度扰动以测试水阻力）
        # "reset_root_state": EventTermCfg(
        #     func=mdp.reset_root_state_uniform,
        #     mode="reset",
        #     params={
        #         "pose_range": {},  # 位置不扰动，使用初始状态
        #         "velocity_range": {
        #             "x": (-0.5, 0.5),
        #             "y": (-0.5, 0.5),
        #             "z": (-0.3, 0.3),
        #         },
        #     },
        # ),
        # 关节重置
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                # 使用正则表达式匹配所有的舵机关节(包含上下左右)
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_servo_.*",)),
            },
        ),
        # 水动力学（浮力 + 水阻力）
        "hydrodynamic_forces": EventTermCfg(
            func=mdp.apply_hydrodynamic_forces,
            mode="step",
            params={
                "water_density": 1000.0,
                "water_viscosity": 0.0009,
                "mass": AUV_MASS,
                "inertia": tuple(AUV_INERTIA),
                "max_force_limit": 5000.0,
                "max_torque_limit": 1000.0,
                "debug": True,
            },
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
            nconmax=2048,
            njmax=4096,
            contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
    )