"""
计算刚体上的水动力和扭矩

基于MuJoCo水动力模型描述实现：https://mujoco.readthedocs.io/en/3.0.1/computation/fluid.html

作者：Ethan Fahnestock 和 Levi "Veevee" Cai (cail@mit.edu)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Sequence
from mjlab.utils.lab_api.math import (
    quat_conjugate, 
    quat_apply, 
    quat_apply_inverse,  
    matrix_from_quat      
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
import numpy as np 
import torch 

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

@dataclass
class HydrodynamicForceModels:
    """水动力模型计算类"""
    num_envs: int  # 环境数量（并行计算的环境数）
    device: torch.device  # 计算设备（CPU/GPU）
    debug: bool = False  # 调试模式开关

    def calculate_buoyancy_forces(self,
                                root_quats_w: torch.Tensor,  # 世界坐标系中的机器人朝向（四元数）
                                fluid_density: float,  # 流体密度 (kg/m³)
                                volumes: torch.Tensor,  # 刚体体积 (m³)
                                g_mag: float,  # 重力加速度大小 (m/s²)
                                com_to_cob_offsets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算完全浸没在流体中的刚体受到的浮力作用（力和扭矩）
        返回的力和扭矩都在刚体自身的坐标系中
        
        参数：
            root_quats_w: 世界坐标系中的四元数姿态 [num_envs, 4]
            fluid_density: 流体密度标量
            volumes: 每个环境的刚体体积 [num_envs, 1]
            g_mag: 重力大小
            com_to_cob_offsets: 质心(COM)到浮心(COB)的偏移量 [num_envs, 3]
            
        返回：
            (浮力向量, 浮力扭矩) 元组，都在刚体坐标系中
        """

        # 初始化浮力和扭矩张量
        buoyancy_forces_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        buoyancy_torques_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # 在世界坐标系中创建浮力方向（与重力方向相反）
        buoyancy_directions_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        buoyancy_directions_w[..., 2] = 1.0  # Z轴向上（对抗重力方向）
        
        # 调试信息输出
        if self.debug: 
            print(f"root_quats shape: {root_quats_w.shape}, buoyancy_vectors shape: {buoyancy_directions_w.shape}")

        # 将浮力方向转换到刚体坐标系
        # 使用四元数将世界系向量转换到本体系
        buoyancy_directions_b = quat_apply(root_quats_w, buoyancy_directions_w)

        # 计算在浮心处的浮力（阿基米德原理）
        # 浮力 = 密度 * 体积 * 重力加速度 * 方向
        buoyancy_forces_at_cob_b = buoyancy_directions_b * fluid_density * volumes.repeat(1,3) * g_mag
        
        # 浮力向量（在刚体坐标系中）
        buoyancy_forces_b = buoyancy_forces_at_cob_b

        # 计算浮力扭矩：τ = r × F
        # 其中r是质心到浮心的偏移向量
        buoyancy_torques_b = torch.cross(com_to_cob_offsets, buoyancy_forces_at_cob_b, dim=-1)

        # 调试输出
        if self.debug: 
            print(f"计算浮力值: 力为 {buoyancy_forces_b}, 扭矩为 {buoyancy_torques_b}")

        return (buoyancy_forces_b, buoyancy_torques_b)
  
    def _calculate_inferred_half_dimensions(self, inertias, masses):
        """
        计算车辆"等效惯性盒"的推断半尺寸
        
        根据MuJoCo模型，将车辆近似为长方体，通过惯性张量推算等效尺寸
        公式：r_i = √[3/(2M) * (I_jj + I_kk - I_ii)]
        
        参数：
            inertias: 惯性张量 [num_envs, 3] (I_xx, I_yy, I_zz)
            masses: 质量 [num_envs, 1]
            
        返回：
            等效半尺寸 [num_envs, 3] (r_x, r_y, r_z)
        """
        # 计算等效半尺寸
        # 注意：torch.roll用于循环获取I_jj和I_kk分量
        term = (torch.roll(inertias, 1, 1) +  # 循环移位获取下一个分量
                torch.roll(inertias, -1, 1) -  # 循环移位获取前一个分量
                inertias)  # 当前分量
        
        # 防止数值问题：确保项为正
        term = torch.clamp(term, min=1e-6)
        
        r = torch.sqrt((3/(2 * masses.repeat(1,3))) * term)
        
        # 限制最小尺寸，避免除零
        r = torch.clamp(r, min=0.01)
        
        return r

    def calculate_quadratic_drag_forces(self,
                                   root_linvels_b: torch.Tensor,  # 本体系中的线速度
                                   root_angvels_b: torch.Tensor,  # 本体系中的角速度
                                   inertias: torch.Tensor,        # 惯性张量
                                   masses: torch.Tensor,          # 质量
                                   fluid_density_rho: float       # 流体密度
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算二次阻力（惯性阻力）和扭矩
        
        基于等效长方体模型：
        f_D,i = -2ρ * r_j * r_k * |v_i| * v_i
        g_D,i = -0.5ρ * r_i * (r_j⁴ + r_k⁴) * |ω_i| * ω_i
        
        参数：
            所有输入都在刚体自身坐标系中
            root_linvels_b: 线速度 [num_envs, 3]
            root_angvels_b: 角速度 [num_envs, 3]
            inertias: 惯性张量 [num_envs, 3]
            masses: 质量 [num_envs, 1]
            fluid_density_rho: 流体密度
            
        返回：
            (阻力向量, 阻力扭矩) 元组
        """

        # 计算等效半尺寸
        ri = self._calculate_inferred_half_dimensions(inertias, masses)
        # 限制速度范围防止数值爆炸
        max_linvel = 10.0  # m/s
        max_angvel = 10.0  # rad/s
        root_linvels_b = torch.clamp(root_linvels_b, -max_linvel, max_linvel)
        root_angvels_b = torch.clamp(root_angvels_b, -max_angvel, max_angvel)
        # 获取旋转后的分量 (r_j, r_k)
        rj = torch.roll(ri, 1, 1)  # 循环移位获取下一个分量
        rk = torch.roll(ri, -1, 1) # 循环移位获取前一个分量

        # 计算线速度阻力
        drag_coefficient = 0.1
        forces = drag_coefficient * -2. * fluid_density_rho * rj * rk * torch.abs(root_linvels_b) * root_linvels_b
        
        # 计算角速度阻力扭矩
        torques = drag_coefficient * -0.5 * fluid_density_rho * ri * (torch.pow(rj,4) + torch.pow(rk,4)) * torch.abs(root_angvels_b) * root_angvels_b

        return (forces, torques)

    def calculate_linear_viscous_forces(self, 
                                       root_linvels_b: torch.Tensor,  # 本体系线速度
                                       root_angvels_b: torch.Tensor,  # 本体系角速度
                                       inertias: torch.Tensor,        # 惯性张量
                                       masses: torch.Tensor,          # 质量
                                       fluid_viscosity_beta: float    # 流体粘度
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算线性粘性力（斯托克斯阻力）和扭矩
        
        基于等效球体模型：
        f_V,i = -6βπ * r_eq * v_i
        g_V,i = -8βπ * r_eq³ * ω_i
        
        参数：
            所有输入都在刚体自身坐标系中
            fluid_viscosity_beta: 流体动力粘度
            
        返回：
            (粘性力向量, 粘性扭矩) 元组
        """
        # 计算等效半尺寸
        ri = self._calculate_inferred_half_dimensions(inertias, masses)
        # 限制速度范围防止数值爆炸
        max_linvel = 10.0  # m/s
        max_angvel = 10.0  # rad/s
        root_linvels_b = torch.clamp(root_linvels_b, -max_linvel, max_linvel)
        root_angvels_b = torch.clamp(root_angvels_b, -max_angvel, max_angvel)
        # 计算等效球体半径（三个方向的平均值）
        r_eq = torch.mean(ri, 1, keepdim=True)
        # 扩展维度以便广播计算
        r_eq = r_eq.repeat(1,3)
        
        # 计算线速度粘性力
        viscous_coefficient = 0.1
        forces = viscous_coefficient * -6. * fluid_viscosity_beta * torch.pi * r_eq * root_linvels_b
        # 计算角速度粘性扭矩
        torques = viscous_coefficient * -8. * fluid_viscosity_beta * torch.pi * torch.pow(r_eq, 3) * root_angvels_b
        
        return (forces, torques)

    def calculate_density_and_viscosity_forces(self, 
                                             root_quats_w: torch.Tensor,  # 世界系姿态
                                             root_linvels_w: torch.Tensor,  # 世界系线速度 [num_envs, 3]
                                             root_angvels_w: torch.Tensor,  # 世界系角速度 [num_envs, 3]
                                             inertias: torch.Tensor,  # 惯性张量 [num_envs, 3]
                                             water_beta: float,  # 水粘度
                                             water_rho: float,  # 水密度
                                             masses: torch.Tensor  # 质量
                                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算总水动力（二次阻力+粘性力）和扭矩
        
        步骤：
        1. 将速度从世界坐标系转换到刚体坐标系
        2. 分别计算二次阻力和粘性力
        3. 返回两种力的组合
        
        返回：
            (f_d, g_d, f_v, g_v) 元组
            f_d: 二次阻力 [num_envs, 3]
            g_d: 二次阻力扭矩 [num_envs, 3]
            f_v: 粘性力 [num_envs, 3]
            g_v: 粘性扭矩 [num_envs, 3]
        """

        # 将速度从世界坐标系转换到刚体坐标系
        # 使用与浮力计算相同的转换：quat_apply(root_quats_w, v_world)
        root_linvels_b = quat_apply(root_quats_w, root_linvels_w)
        # 将角速度转换到本体系
        root_angvels_b = quat_apply(root_quats_w, root_angvels_w)
  
        # 计算二次阻力（密度相关）
        f_d, g_d = self.calculate_quadratic_drag_forces(root_linvels_b, root_angvels_b, inertias, masses, water_rho)
        # 计算粘性力（粘度相关）
        f_v, g_v = self.calculate_linear_viscous_forces(root_linvels_b, root_angvels_b, inertias, masses, water_beta)
        
        return (f_d, g_d, f_v, g_v)


def apply_hydrodynamic_forces(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    water_density: float = 997.0,
    water_viscosity: float = 0.001306,
    volume: float = 0.023,
    com_to_cob_offset: tuple[float, float, float] = (0.0, 0.0, 0.3),
    mass: float = 100.0,  # 默认质量 (kg)
    inertia: tuple[float, float, float] = (19.3, 19.3, 1.125),  # 默认惯性 [I_xx, I_yy, I_zz] (kg·m²)
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    debug: bool = False
) -> None:
    """应用水动力和扭矩到机器人
    
    这是一个MDP事件函数，可以在环境配置中注册为事件。
    需要在实际使用前完善力施加的具体实现。
    """
    # 获取资产（机器人）
    asset = env.scene[asset_cfg.name]
    
    # 获取机器人状态
    root_quats_w = asset.data.root_link_quat_w
    root_linvels_w = asset.data.root_link_lin_vel_w
    root_angvels_w = asset.data.root_link_ang_vel_w
    # 替换NaN为零
    root_quats_w = torch.where(torch.isnan(root_quats_w), torch.zeros_like(root_quats_w), root_quats_w)
    root_linvels_w = torch.where(torch.isnan(root_linvels_w), torch.zeros_like(root_linvels_w), root_linvels_w)
    root_angvels_w = torch.where(torch.isnan(root_angvels_w), torch.zeros_like(root_angvels_w), root_angvels_w)
    
    # 使用传入的质量和惯性参数
    num_envs = env.num_envs
    device = torch.device(env.device) if isinstance(env.device, str) else env.device
    
    masses = torch.tensor([mass], dtype=torch.float, device=device).reshape(1,1).repeat(num_envs, 1)
    inertias = torch.tensor([inertia], dtype=torch.float, device=device).reshape(1,3).repeat(num_envs, 1)
    
    # 创建水动力模型
    force_model = HydrodynamicForceModels(num_envs, device, debug=debug)  # type: ignore
    
    # 准备体积张量
    volumes = torch.tensor([volume], dtype=torch.float, device=device).reshape(1,1).repeat(num_envs, 1)
    com_to_cob_offsets = torch.tensor([com_to_cob_offset], dtype=torch.float, device=device).reshape(1,3).repeat(num_envs, 1)
    
    # 计算浮力
    buoyancy_force, buoyancy_torque = force_model.calculate_buoyancy_forces(
        root_quats_w, water_density, volumes, 9.81, com_to_cob_offsets
    )
    
    # 计算阻力和粘性力
    f_d, g_d, f_v, g_v = force_model.calculate_density_and_viscosity_forces(
        root_quats_w, root_linvels_w, root_angvels_w,
        inertias, water_viscosity, water_density, masses
    )
    
    # 总力和扭矩
    total_force = buoyancy_force + f_d + f_v
    total_torque = buoyancy_torque + g_d + g_v
    
    # 检查NaN并打印调试信息
    if torch.isnan(total_force).any() or torch.isnan(total_torque).any():
        print(f"[Hydrodynamics] NaN detected in forces/torques")
        print(f"  buoyancy_force: {buoyancy_force}")
        print(f"  f_d: {f_d}")
        print(f"  f_v: {f_v}")
        print(f"  root_linvels_w: {root_linvels_w}")
        print(f"  root_angvels_w: {root_angvels_w}")
        # 将NaN替换为零
        total_force = torch.where(torch.isnan(total_force), torch.zeros_like(total_force), total_force)
        total_torque = torch.where(torch.isnan(total_torque), torch.zeros_like(total_torque), total_torque)
    
    # 施加外力到机器人
    # 限制力与扭矩的大小，防止数值爆炸
    max_force = 5000.0  # N
    max_torque = 1000.0  # N·m
    total_force = torch.clamp(total_force, -max_force, max_force)
    total_torque = torch.clamp(total_torque, -max_torque, max_torque)
    # 将力和扭矩从刚体坐标系转换到世界坐标系
    total_force_w = quat_apply(root_quats_w, total_force)
    total_torque_w = quat_apply(root_quats_w, total_torque)
    # 添加重力（重量）力（世界坐标系，方向向下）
    g_mag = 9.81
    weight_force_w = torch.zeros_like(total_force_w)
    weight_force_w[:, 2] = -masses.squeeze(1) * g_mag
    total_force_w += weight_force_w
    
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)
    
    # 获取机器人实体
    asset = env.scene[asset_cfg.name]
    # 确定应用力的身体索引（默认为根身体，索引0）
    body_ids = asset_cfg.body_ids
    if body_ids is None:
        body_ids = [0]
    
    # 计算身体数量
    if isinstance(body_ids, slice):
        # 如果是slice(None)，应用所有身体
        num_bodies = asset.data.body_link_pos_w.shape[1]
    else:
        num_bodies = len(body_ids)  # type: ignore
    
    # 扩展维度以匹配 (num_envs, num_bodies, 3)
    total_force_w_expanded = total_force_w.unsqueeze(1).repeat(1, num_bodies, 1)
    total_torque_w_expanded = total_torque_w.unsqueeze(1).repeat(1, num_bodies, 1)
    
    # 施加外力到仿真
    asset.write_external_wrench_to_sim(
        forces=total_force_w_expanded,
        torques=total_torque_w_expanded,
        env_ids=env_ids,
        body_ids=body_ids
    )
    
    # 调试信息
    if force_model.debug:
        print(f"水动力计算完成: 力={total_force.mean(dim=0)}, 扭矩={total_torque.mean(dim=0)}")
        print(f"世界坐标系力={total_force_w.mean(dim=0)}, 扭矩={total_torque_w.mean(dim=0)}")


def apply_specific_force_torque(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    forces: torch.Tensor,  # [num_envs, 3]
    torques: torch.Tensor, # [num_envs, 3]
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> None:
    """施加特定的力和扭矩到机器人
    
    这是一个辅助函数，用于施加计算得到的水动力。
    假设输入的力和扭矩已经在世界坐标系中。
    """
    # 获取机器人实体
    asset = env.scene[asset_cfg.name]
    # 确定应用力的身体索引（默认为根身体，索引0）
    body_ids = asset_cfg.body_ids
    if body_ids is None:
        body_ids = [0]
    
    # 计算身体数量
    if isinstance(body_ids, slice):
        # 如果是slice(None)，应用所有身体
        num_bodies = asset.data.body_link_pos_w.shape[1]
    else:
        # body_ids is Sequence[int]
        num_bodies = len(body_ids)  # type: ignore
    
    # 扩展维度以匹配 (num_envs, num_bodies, 3)
    forces_expanded = forces.unsqueeze(1).repeat(1, num_bodies, 1)
    torques_expanded = torques.unsqueeze(1).repeat(1, num_bodies, 1)
    
    # 施加外力到仿真
    asset.write_external_wrench_to_sim(
        forces=forces_expanded,
        torques=torques_expanded,
        env_ids=env_ids,
        body_ids=body_ids
    )


def apply_thruster_forces(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    action_name: str = "thrusters",  # 动作管理器中的推进器动作名称
    max_thrust: float = 100.0,  # 最大推力（牛顿）
    debug: bool = False
) -> None:
    """计算并施加推进器推力到机器人base
    
    基于四个推进器的姿态和动作值，计算推进器推力并施加到机器人base上。
    推进器推力方向为推进器body局部坐标系的-Z轴（假设推进器向前推）。
    
    参数:
        env: RL环境
        env_ids: 环境ID
        thrust_actions: 推进器动作值 [num_envs, 4]，顺序为 [up, down, left, right]
        asset_cfg: 资产配置
        debug: 调试模式
    """
    # 获取机器人实体
    asset = env.scene[asset_cfg.name]
    
    # 改为使用舵机body（推力作用点）
    thruster_body_names = [
        "body_servo_up",      # 上舵机位置
        "body_servo_down",    # 下舵机位置  
        "body_servo_left",    # 左舵机位置
        "body_servo_right",   # 右舵机位置
    ]
    
    # 获取body索引
    body_indices = []
    for body_name in thruster_body_names:
        # 查找body索引
        try:
            idx = asset.spec.body_names.index(body_name)
            body_indices.append(idx)
        except ValueError:
            if debug:
                print(f"警告: 未找到body '{body_name}'，可用body: {asset.spec.body_names}")
            # 如果找不到，尝试使用默认索引（假设顺序相同）
            pass
    
    # 如果通过名称找不到，使用假设的索引（0, 1, 2, 3）
    if len(body_indices) != 4:
        if debug:
            print(f"使用默认body索引 [0, 1, 2, 3]")
        body_indices = [0, 1, 2, 3]  # 假设四个推进器body是前四个
    
    num_envs = env.num_envs
    device = torch.device(env.device) if isinstance(env.device, str) else env.device
    
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)
    
    # 获取推进器body的姿态（世界坐标系）
    # body_link_pos_w: [num_envs, num_bodies, 3]
    # body_link_quat_w: [num_envs, num_bodies, 4]
    thruster_pos_w = asset.data.body_link_pos_w[:, body_indices, :]  # [num_envs, 4, 3]
    thruster_quat_w = asset.data.body_link_quat_w[:, body_indices, :]  # [num_envs, 4, 4]
    
    # 从动作管理器获取推进器动作值
    # 假设推进器动作是完整动作向量的最后4个维度
    try:
        # 获取完整动作向量
        # 注意: env.action_manager.action 应该在process_action之后包含当前动作
        full_action = env.action_manager.action
        if full_action is None:
            if debug:
                print("警告: 动作管理器中没有动作值，使用零动作")
            thrust_action_values = torch.zeros((num_envs, 4), device=device)
        else:
            # 获取动作维度
            action_dim = full_action.shape[1]
            if action_dim >= 4:
                # 取最后4个元素作为推进器动作
                thrust_action_values = full_action[:, -4:]
            else:
                if debug:
                    print(f"错误: 动作维度 {action_dim} 小于4，无法提取推进器动作")
                thrust_action_values = torch.zeros((num_envs, 4), device=device)
    except Exception as e:
        if debug:
            print(f"获取推进器动作值时出错: {e}, 使用零动作")
        thrust_action_values = torch.zeros((num_envs, 4), device=device)
    
    # 推进器推力系数（将动作值转换为牛顿）
    # 动作值范围假设为 [-1, 1]，对应推力 [-max_thrust, max_thrust]
    thrust_magnitudes = thrust_action_values * max_thrust  # [num_envs, 4]
    
    # 推进器推力方向（推进器局部坐标系中的方向）
    # 假设推进器推力沿着局部-Z轴方向（向前推）
    local_thrust_direction = torch.tensor([0.0, 0.0, -1.0], device=device)  # 局部-Z轴
    local_thrust_direction = local_thrust_direction.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
    
    # 初始化总力和扭矩
    total_force_w = torch.zeros((num_envs, 3), device=device)
    total_torque_w = torch.zeros((num_envs, 3), device=device)
    
    # 对每个推进器计算力和扭矩
    for i in range(4):
        # 获取当前推进器的四元数
        quat_w = thruster_quat_w[:, i, :]  # [num_envs, 4]
        
        # 将推力方向从局部坐标系转换到世界坐标系
        thrust_dir_w = quat_apply(quat_w, local_thrust_direction)  # [num_envs, 3]
        
        # 计算推力向量（世界坐标系）
        thrust_mag = thrust_magnitudes[:, i].unsqueeze(1)  # [num_envs, 1]
        thrust_vector_w = thrust_dir_w * thrust_mag  # [num_envs, 3]
        
        # 获取推力作用点（世界坐标系）
        thrust_point_w = thruster_pos_w[:, i, :]  # [num_envs, 3]
        
        # 获取base位置（假设base是第一个body）
        base_pos_w = asset.data.body_link_pos_w[:, 0, :]  # [num_envs, 3]
        
        # 计算相对于base的力臂
        r = thrust_point_w - base_pos_w  # [num_envs, 3]
        
        # 计算扭矩：τ = r × F
        # 使用叉积计算扭矩
        torque_w = torch.cross(r, thrust_vector_w, dim=1)  # [num_envs, 3]
        
        # 累加到总力和扭矩
        total_force_w += thrust_vector_w
        total_torque_w += torque_w
        
        if debug and i == 0 and env_ids[0] == 0:
            print(f"推进器 {i}: 动作值={thrust_action_values[0, i]:.3f}, 推力大小={thrust_mag[0, 0]:.1f} N")
            print(f"  局部推力方向: {local_thrust_direction[0, 0].cpu().numpy()}")
            print(f"  世界推力方向: {thrust_dir_w[0].cpu().numpy()}")
            print(f"  推力点位置: {thrust_point_w[0].cpu().numpy()}")
            print(f"  力臂 r: {r[0].cpu().numpy()}")
    
    if debug and env_ids[0] == 0:
        print(f"总推进器力: {total_force_w[0].cpu().numpy()}")
        print(f"总推进器扭矩: {total_torque_w[0].cpu().numpy()}")
    
    # 施加力和扭矩到base
    # 扩展维度以匹配 (num_envs, num_bodies, 3)
    total_force_w_expanded = total_force_w.unsqueeze(1)  # [num_envs, 1, 3]
    total_torque_w_expanded = total_torque_w.unsqueeze(1)  # [num_envs, 1, 3]
    
    # 施加到base（索引0）
    asset.write_external_wrench_to_sim(
        forces=total_force_w_expanded,
        torques=total_torque_w_expanded,
        env_ids=env_ids,
        body_ids=[0]  # 施加到base
    )


if __name__ == "__main__":
    """单元测试"""
    # 设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 物理参数
    water_rho = 997.0  # 水密度 (kg/m³)
    water_beta = 0.001306  # 水粘度 (Pa·s) @ 50°F
    g_mag = 9.81  # 重力加速度 (m/s²)
    num_envs = 4  # 测试环境数量
    com_to_cob_offset = torch.tensor([0.0, 0.0, 0.3], dtype=torch.float, device=device, requires_grad=False).reshape(1,3).repeat(num_envs, 1) 
    volume = 1.0  # 刚体体积 (m³)，中性浮力
    volume_tensor = torch.tensor([volume], dtype=torch.float, device=device).reshape(1,1).repeat(num_envs, 1)

    # 创建水动力模型实例
    forceModel = HydrodynamicForceModels(num_envs, device, True)

    # 测试浮力计算的四元数（注意：这里使用[w, x, y, z]格式）
    # 使用标准旋转四元数
    root_quats = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # 无旋转（单位四元数）
        [0.7071068, 0.7071068, 0.0, 0.0],  # 绕X轴90度
        [0.7071068, 0.0, 0.7071068, 0.0],  # 绕Y轴90度
        [0.7071068, 0.0, 0.0, 0.7071068],  # 绕Z轴90度
    ]).to(device)

    # 预期的浮力结果（在刚体坐标系中）
    # 基于标准旋转计算
    buoyancy_magnitude = volume * water_rho * g_mag
    true_b_forces = torch.tensor([
        [0.0, 0.0, buoyancy_magnitude],  # 无旋转：浮力向上 [0,0,1]
        [0.0, -buoyancy_magnitude, 0.0],  # X旋转90°：浮力在-Y方向 [0,-1,0]
        [buoyancy_magnitude, 0.0, 0.0],  # Y旋转90°：浮力在+X方向 [1,0,0]
        [0.0, 0.0, buoyancy_magnitude],  # Z旋转90°：浮力仍向上 [0,0,1]（绕Z轴旋转不影响Z方向）
    ]).to(device)

    # 预期的浮力扭矩结果 τ = r × F
    # r = [0, 0, 0.3] (com_to_cob_offset)
    torque_magnitude = 0.3 * water_rho * g_mag * volume
    true_b_torques = torch.tensor([
        [0.0, 0.0, 0.0],  # 无旋转：r × [0,0,F] = [0,0,0]
        [torque_magnitude, 0.0, 0.0],  # X旋转90°：r × [0,-F,0] = [0.3*F, 0, 0]
        [0.0, torque_magnitude, 0.0],  # Y旋转90°：r × [F,0,0] = [0, 0.3*F, 0]
        [0.0, 0.0, 0.0],  # Z旋转90°：r × [0,0,F] = [0,0,0]
    ]).to(device)

    # 计算浮力
    b_force, b_torque = forceModel.calculate_buoyancy_forces(root_quats, water_rho, volume_tensor, g_mag, com_to_cob_offset)

    # 验证浮力计算结果（使用混合误差检查）
    b_force_np = b_force.cpu().numpy()
    true_b_forces_np = true_b_forces.cpu().numpy()
    
    # 计算绝对误差
    force_abs_error = np.abs(b_force_np - true_b_forces_np)
    
    # 对于期望值接近0的情况，使用绝对误差；否则使用相对误差
    zero_threshold = 1e-6
    force_errors = np.zeros_like(force_abs_error)
    
    for i in range(force_abs_error.shape[0]):
        for j in range(force_abs_error.shape[1]):
            if np.abs(true_b_forces_np[i, j]) > zero_threshold:
                # 相对误差（百分比）
                force_errors[i, j] = force_abs_error[i, j] / np.abs(true_b_forces_np[i, j])
            else:
                # 绝对误差（当期望值为0时）
                force_errors[i, j] = force_abs_error[i, j]
    
    # 不同的容差：相对误差使用1e-4，绝对误差使用1e-3
    max_error = force_errors.max()
    if max_error > 1e-4:  # 检查是否超过相对容差
        # 但对于期望值为0的情况，使用更宽松的绝对容差
        zero_mask = np.abs(true_b_forces_np) <= zero_threshold
        if np.any(zero_mask):
            # 检查期望值为0的情况的绝对误差
            zero_abs_errors = force_abs_error[zero_mask]
            if zero_abs_errors.max() > 1e-3:  # 绝对容差1e-3
                print(f"浮力计算错误: 最大误差 {max_error:.2e}")
                print(f"期望值为0处的最大绝对误差: {zero_abs_errors.max():.6e}")
                print(f"计算值:\n {b_force}")
                print(f"期望值:\n {true_b_forces}")
            else:
                # 期望值为0的情况通过，检查非零值的相对误差
                non_zero_mask = np.abs(true_b_forces_np) > zero_threshold
                if np.any(non_zero_mask):
                    non_zero_rel_errors = force_errors[non_zero_mask]
                    if non_zero_rel_errors.max() > 1e-4:
                        print(f"浮力计算错误: 非零期望值处最大相对误差 {non_zero_rel_errors.max():.2%}")
                        print(f"计算值:\n {b_force}")
                        print(f"期望值:\n {true_b_forces}")
                    else:
                        print(f"浮力测试通过! 非零值最大相对误差: {non_zero_rel_errors.max():.2%}")
                else:
                    print(f"浮力测试通过! 最大绝对误差: {zero_abs_errors.max():.6e}")
        else:
            print(f"浮力计算错误: 最大相对误差 {max_error:.2%}")
            print(f"计算值:\n {b_force}")
            print(f"期望值:\n {true_b_forces}")
    else:
        print(f"浮力测试通过! 最大误差: {max_error:.2e}")

    # 验证扭矩计算结果
    b_torque_np = b_torque.cpu().numpy()
    true_b_torques_np = true_b_torques.cpu().numpy()
    
    torque_abs_error = np.abs(b_torque_np - true_b_torques_np)
    
    torque_errors = np.zeros_like(torque_abs_error)
    for i in range(torque_abs_error.shape[0]):
        for j in range(torque_abs_error.shape[1]):
            if np.abs(true_b_torques_np[i, j]) > zero_threshold:
                torque_errors[i, j] = torque_abs_error[i, j] / np.abs(true_b_torques_np[i, j])
            else:
                torque_errors[i, j] = torque_abs_error[i, j]
    
    max_torque_error = torque_errors.max()
    if max_torque_error > 1e-4:
        zero_mask = np.abs(true_b_torques_np) <= zero_threshold
        if np.any(zero_mask):
            zero_abs_errors = torque_abs_error[zero_mask]
            if zero_abs_errors.max() > 1e-3:
                print(f"扭矩计算错误: 最大误差 {max_torque_error:.2e}")
                print(f"期望值为0处的最大绝对误差: {zero_abs_errors.max():.6e}")
                print(f"计算值:\n {b_torque}")
                print(f"期望值:\n {true_b_torques}")
            else:
                non_zero_mask = np.abs(true_b_torques_np) > zero_threshold
                if np.any(non_zero_mask):
                    non_zero_rel_errors = torque_errors[non_zero_mask]
                    if non_zero_rel_errors.max() > 1e-4:
                        print(f"扭矩计算错误: 非零期望值处最大相对误差 {non_zero_rel_errors.max():.2%}")
                        print(f"计算值:\n {b_torque}")
                        print(f"期望值:\n {true_b_torques}")
                    else:
                        print(f"扭矩测试通过! 非零值最大相对误差: {non_zero_rel_errors.max():.2%}")
                else:
                    print(f"扭矩测试通过! 最大绝对误差: {zero_abs_errors.max():.6e}")
        else:
            print(f"扭矩计算错误: 最大相对误差 {max_torque_error:.2%}")
            print(f"计算值:\n {b_torque}")
            print(f"期望值:\n {true_b_torques}")
    else:
        print(f"扭矩测试通过! 最大误差: {max_torque_error:.2e}")

    # 以下是阻力测试的框架（实际测试需要提供惯性张量）
    root_linvels = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], device=device).unsqueeze(0)  # 添加批次维度
    
    root_angvels = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], device=device).unsqueeze(0)
    
    # 实际测试时需要提供真实的惯性张量
    # inertias = ... 
    # f_d, g_d, f_v, g_v = forceModel.calculate_density_and_viscosity_forces(...)