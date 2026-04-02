from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def align_z_with_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.5,  # 用于指数形式的宽度参数
) -> torch.Tensor:
    """奖励机器人的z轴指向期望速度方向
    
    参数:
        mode: 奖励计算模式
            - "dot_product": 直接使用点积 [-1, 1]
            - "angle": 使用归一化角度 [0, 1]
            - "exponential": 使用指数形式 (0, 1]
    """
    # 获取资产和命令
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    vel_command_b = command[:, :]   # [num_envs, 3]   
    vel_norm = torch.norm(vel_command_b, dim=1, keepdim=True)
    
    # 处理零速度情况
    vel_dir_b = torch.where(
        vel_norm > 1e-6,
        vel_command_b / vel_norm,
        torch.zeros_like(vel_command_b)
    )
    
    # 将速度方向转换到世界坐标系
    root_quat_w = asset.data.root_link_quat_w
    vel_dir_w = quat_apply(root_quat_w, vel_dir_b)
    
    # 获取z轴在世界坐标系中的方向
    z_axis_b = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    z_axis_w = quat_apply(root_quat_w, z_axis_b)
    
    # 计算对齐度
    alignment = torch.sum(z_axis_w * vel_dir_w, dim=1)  # 点积

    angle = torch.acos(torch.clamp(alignment, -0.9999, 0.9999))
    return torch.exp(-angle**2 / (2 * std**2))  # (0, 1]