from __future__ import annotations

import torch
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def allocate_thruster_forces(
    actions: torch.Tensor,
    max_servo_angle: float,
    max_thrust: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 6 维策略动作映射为：
    - 2 维舵机角目标
    - 8 维推进器分量力

    动作:
        actions[:, 0] -> theta12_norm
        actions[:, 1] -> theta34_norm
        actions[:, 2] -> F1_norm
        actions[:, 3] -> F2_norm
        actions[:, 4] -> F3_norm
        actions[:, 5] -> F4_norm

    返回:
        servo_targets: [num_envs, 2]
        thrust_targets: [num_envs, 8]
    """
    # 舵机角（确保数值稳定）
    theta12 = actions[:, 0].clamp(-1.0, 1.0) * max_servo_angle
    theta34 = actions[:, 1].clamp(-1.0, 1.0) * max_servo_angle

    # 推力（确保数值稳定）
    F1 = actions[:, 2].clamp(-1.0, 1.0) * max_thrust
    F2 = actions[:, 3].clamp(-1.0, 1.0) * max_thrust
    F3 = actions[:, 4].clamp(-1.0, 1.0) * max_thrust
    F4 = actions[:, 5].clamp(-1.0, 1.0) * max_thrust

    sin12 = torch.sin(theta12)
    cos12 = torch.cos(theta12)
    sin34 = torch.sin(theta34)
    cos34 = torch.cos(theta34)

    # 舵机输出
    servo_targets = torch.stack([theta12, theta34], dim=-1)

    # 推进器分量（顺序需与 actuator_names 对齐）
    thrust_targets = torch.stack(
        [
            F1 * cos12,      # up_z
            F1 * sin12,      # up_y
            F2 * cos12,      # down_z
            F2 * sin12,      # down_y
            F3 * cos34,      # left_z
            F3 * sin34,      # left_x
            F4 * cos34,      # right_z
            F4 * sin34,      # right_x
        ],
        dim=-1,
    )

    return servo_targets, thrust_targets


class AuvThrusterAllocationAction(ActionTerm):
    """6维动作 -> thruster_allocation -> 2舵机 + 8路等效力"""

    cfg: "AuvThrusterAllocationActionCfg"

    def __init__(self, cfg: "AuvThrusterAllocationActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)

        # 获取舵机和推进器执行器ID
        self._servo_actuator_ids = self._entity.find_actuators(cfg.servo_joint_names)[0]      # 形状 (2,)
        self._thruster_actuator_ids = self._entity.find_actuators(cfg.thruster_joint_names)[0]  # 形状 (8,)

        # 存储原始动作和处理后的目标
        self._raw_actions = torch.zeros(env.num_envs, 6, device=env.device)
        self._servo_targets = torch.zeros(env.num_envs, 2, device=env.device)    # 舵机角目标
        self._thruster_targets = torch.zeros(env.num_envs, 8, device=env.device)  # 推进器力目标

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # 计算舵机和推进器目标
        servo_targets, thrust_targets = allocate_thruster_forces(
            actions,
            max_servo_angle=self.cfg.max_servo_angle,
            max_thrust=self.cfg.max_thrust,
        )

        # 分别存储
        self._servo_targets[:] = servo_targets
        self._thruster_targets[:] = thrust_targets

    def apply_actions(self):
        # 写入舵机控制
        self._entity.data.write_ctrl(self._servo_targets, self._servo_actuator_ids)
        # 写入推进器控制
        self._entity.data.write_ctrl(self._thruster_targets, self._thruster_actuator_ids)


@dataclass(kw_only=True)
class AuvThrusterAllocationActionCfg(ActionTermCfg):
    """配置类"""
    class_type = AuvThrusterAllocationAction

    entity_name: str = "robot"
    servo_joint_names: list[str] = field(default_factory=list)      # 舵机执行器名称（2个）
    thruster_joint_names: list[str] = field(default_factory=list)   # 推进器执行器名称（8个）
    max_servo_angle: float = math.pi / 2
    max_thrust: float = 10.0

    def build(self, env: "ManagerBasedRlEnv") -> "AuvThrusterAllocationAction":
        return AuvThrusterAllocationAction(self, env)