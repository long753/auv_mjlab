"""
计算刚体上的水动力和扭矩

基于MuJoCo水动力模型描述实现：https://mujoco.readthedocs.io/en/3.0.1/computation/fluid.html

作者：Ethan Fahnestock 和 Levi "Veevee" Cai (cail@mit.edu)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple
from mjlab.utils.lab_api.math import (
    quat_apply, 
    quat_apply_inverse,  
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
        # 使用四元数逆变换将世界系向量转换到本体系 (world → body)
        buoyancy_directions_b = quat_apply_inverse(root_quats_w, buoyancy_directions_w)

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

        # 将速度从世界坐标系转换到刚体坐标系 (world → body)
        root_linvels_b = quat_apply_inverse(root_quats_w, root_linvels_w)
        # 将角速度转换到本体系
        root_angvels_b = quat_apply_inverse(root_quats_w, root_angvels_w)
  
        # 计算二次阻力（密度相关）
        f_d, g_d = self.calculate_quadratic_drag_forces(root_linvels_b, root_angvels_b, inertias, masses, water_rho)
        # 计算粘性力（粘度相关）
        f_v, g_v = self.calculate_linear_viscous_forces(root_linvels_b, root_angvels_b, inertias, masses, water_beta)
        
        return (f_d, g_d, f_v, g_v)


# ---------- 模块级缓存，避免每步重复创建张量和模型实例 ----------
_hydro_cache: dict = {}


def _get_or_create_cache(
    num_envs: int,
    device: torch.device,
    mass: float,
    inertia: tuple[float, float, float],
    volume: float,
    com_to_cob_offset: tuple[float, float, float],
    debug: bool,
) -> dict:
    """获取或创建水动力计算所需的缓存张量，避免每步重复分配内存"""
    key = (num_envs, str(device))
    if key not in _hydro_cache:
        _hydro_cache[key] = {
            "model": HydrodynamicForceModels(num_envs, device, debug=debug),
            "masses": torch.tensor([[mass]], dtype=torch.float, device=device).expand(num_envs, 1).clone(),
            "inertias": torch.tensor([list(inertia)], dtype=torch.float, device=device).expand(num_envs, 3).clone(),
            "volumes": torch.tensor([[volume]], dtype=torch.float, device=device).expand(num_envs, 1).clone(),
            "cob_offsets": torch.tensor([list(com_to_cob_offset)], dtype=torch.float, device=device).expand(num_envs, 3).clone(),
            "all_env_ids": torch.arange(num_envs, device=device),
            "body_masses": None,  # 延迟初始化，需要 asset 信息
        }
    return _hydro_cache[key]


def _sanitize_quat(q: torch.Tensor) -> torch.Tensor:
    """归一化四元数，处理 NaN 和零范数的退化情况"""
    # NaN → 0
    q = torch.where(torch.isnan(q), torch.zeros_like(q), q)
    # 计算范数
    norms = q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q = q / norms
    # 范数过小的行（退化四元数）→ 单位四元数 [1,0,0,0]
    degenerate = (norms.squeeze(-1) < 1e-6)
    if degenerate.any():
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
        q[degenerate] = identity
    return q


def _sanitize_vel(v: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    """清理速度张量：NaN → 0，然后 clamp"""
    v = torch.where(torch.isnan(v), torch.zeros_like(v), v)
    return v.clamp(-max_val, max_val)


def _get_body_masses_from_model(asset_cfg_name: str) -> list[float]:
    """从 MuJoCo 模型中提取每个 body 的质量（跳过 world body）。

    返回:
        body 质量列表，顺序与 Entity.body_names / body_ids 一致。
    """
    import mujoco as _mj
    from src.auv_mjlab.assets.robots.auv.auv_constants import get_spec

    spec = get_spec()
    model = spec.compile()
    # Entity 的 body 列表是 spec.bodies[1:]（跳过 world）
    bodies = spec.bodies[1:]
    return [float(model.body_mass[b.id]) for b in bodies]


def apply_hydrodynamic_forces(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    water_density: float = 997.0,
    water_viscosity: float = 0.001306,
    mass: float = 100.0,
    inertia: tuple[float, float, float] = (19.3, 19.3, 1.125),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_force_limit: float = 5000.0,
    max_torque_limit: float = 1000.0,
    debug: bool = False,
) -> None:
    """计算并施加水动力（浮力 + 二次阻力 + 粘性力）到机器人。

    浮力：对每个 body 分别施加 F_i = [0, 0, m_i * g]（世界系竖直向上），
    精确抵消该 body 自身的重力，实现整体中性浮力。
    阻力：仅施加在根 body 上（基于整体质量和惯性计算）。

    作为 EventTermCfg(mode="step") 注册，每个仿真步调用一次。
    所有中间张量通过模块级缓存复用，避免每步分配内存。

    参数:
        env: ManagerBasedRlEnv 实例
        env_ids: 需要施力的环境索引，None 表示全部
        water_density: 水密度 (kg/m³)
        water_viscosity: 水动力粘度 (Pa·s)
        mass: AUV 总质量 (kg)，用于阻力计算
        inertia: 主惯性矩 (I_xx, I_yy, I_zz) (kg·m²)
        asset_cfg: 场景实体配置
        max_force_limit: 力的 clamp 上限 (N)
        max_torque_limit: 扭矩的 clamp 上限 (N·m)
        debug: 是否打印调试信息
    """
    num_envs = env.num_envs
    device = torch.device(env.device) if isinstance(env.device, str) else env.device

    # 1. 获取/创建缓存
    cache = _get_or_create_cache(
        num_envs, device, mass, inertia, 0.0, (0.0, 0.0, 0.0), debug
    )
    hydro_model = cache["model"]

    # 2. 获取机器人实体和状态
    asset = env.scene[asset_cfg.name]
    root_quats_w = _sanitize_quat(asset.data.root_link_quat_w)
    root_linvels_w = _sanitize_vel(asset.data.root_link_lin_vel_w)
    root_angvels_w = _sanitize_vel(asset.data.root_link_ang_vel_w)

    # 3. 计算水阻力（本体系，基于根 body 速度和整体参数）
    f_d, g_d, f_v, g_v = hydro_model.calculate_density_and_viscosity_forces(
        root_quats_w, root_linvels_w, root_angvels_w,
        cache["inertias"], water_viscosity, water_density, cache["masses"]
    )

    # ---- 根 body 阻力处理 ----
    drag_force_b = f_d + f_v
    drag_torque_b = g_d + g_v

    # NaN 安全检查
    nan_mask = torch.isnan(drag_force_b).any(dim=-1) | torch.isnan(drag_torque_b).any(dim=-1)
    if nan_mask.any():
        if debug:
            bad_ids = nan_mask.nonzero(as_tuple=False).squeeze(-1)
            print(f"[Hydrodynamics] NaN in envs {bad_ids.tolist()}, zeroing those envs")
        drag_force_b[nan_mask] = 0.0
        drag_torque_b[nan_mask] = 0.0

    # Clamp 防止数值爆炸
    drag_force_b = drag_force_b.clamp(-max_force_limit, max_force_limit)
    drag_torque_b = drag_torque_b.clamp(-max_torque_limit, max_torque_limit)

    # 本体系 → 世界系
    drag_force_w = quat_apply(root_quats_w, drag_force_b)
    drag_torque_w = quat_apply(root_quats_w, drag_torque_b)

    # ---- 浮力：给每个 body 施加与其自身重力等大反向的力 ----
    # 延迟初始化每个 body 的质量缓存
    if cache["body_masses"] is None:
        body_mass_list = _get_body_masses_from_model(asset_cfg.name)
        cache["body_masses"] = torch.tensor(body_mass_list, dtype=torch.float, device=device)
        cache["num_bodies"] = len(body_mass_list)
        cache["all_body_ids"] = list(range(len(body_mass_list)))
        if debug:
            print(f"[Hydro] 初始化各 body 质量: {body_mass_list}")
            print(f"[Hydro] 总质量: {sum(body_mass_list):.4f} kg, body 数量: {len(body_mass_list)}")

    num_bodies = cache["num_bodies"]
    body_masses = cache["body_masses"]  # [num_bodies]

    g_mag = 9.81
    # 构造每个 body 的浮力：世界系中 [0, 0, m_i * g]
    # 形状: [num_envs, num_bodies, 3]
    buoyancy_forces_w = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
    buoyancy_forces_w[:, :, 2] = body_masses.unsqueeze(0) * g_mag  # 广播到所有环境
    buoyancy_torques_w = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)

    # ---- 合并：根 body 的力 = 阻力 + 浮力，其他 body 只有浮力 ----
    # 将阻力加到根 body（索引 0）的浮力上
    buoyancy_forces_w[:, 0, :] += drag_force_w
    buoyancy_torques_w[:, 0, :] += drag_torque_w

    # 施加到所有 body
    if env_ids is None:
        env_ids = cache["all_env_ids"]

    asset.write_external_wrench_to_sim(
        forces=buoyancy_forces_w,     # [N, num_bodies, 3]
        torques=buoyancy_torques_w,   # [N, num_bodies, 3]
        env_ids=env_ids,
        body_ids=cache["all_body_ids"],
    )

    # 调试输出
    if debug:
        total_buoyancy = (body_masses * g_mag).sum().item()
        mean_drag = drag_force_b.mean(dim=0)
        print(f"[Hydro] 各body浮力总和: {total_buoyancy:.4f} N (应等于总重力), "
              f"根body阻力(本体系): [{mean_drag[0]:.6f}, {mean_drag[1]:.6f}, {mean_drag[2]:.6f}] N")


if __name__ == "__main__":
    """单元测试"""

    def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor,
                     atol: float = 5e-3, rtol: float = 1e-4) -> bool:
        """比较两个张量，使用混合容差（绝对+相对），打印结果"""
        passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        if passed:
            max_diff = (actual - expected).abs().max().item()
            print(f"  ✓ {name} 通过 (最大差值: {max_diff:.2e})")
        else:
            print(f"  ✗ {name} 失败")
            print(f"    计算值:\n{actual}")
            print(f"    期望值:\n{expected}")
            print(f"    差值:\n{actual - expected}")
        return passed

    # ========== 设置 ==========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    water_rho = 997.0       # 水密度 (kg/m³)
    water_beta = 0.001306   # 水粘度 (Pa·s)
    g_mag = 9.81            # 重力加速度 (m/s²)
    num_envs = 4

    com_to_cob_offset = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float, device=device).repeat(num_envs, 1)
    volume = 1.0
    volume_tensor = torch.tensor([[volume]], dtype=torch.float, device=device).repeat(num_envs, 1)
    mass = 100.0
    masses = torch.tensor([[mass]], dtype=torch.float, device=device).repeat(num_envs, 1)
    inertias = torch.tensor([[19.3, 19.3, 1.125]], dtype=torch.float, device=device).repeat(num_envs, 1)

    model = HydrodynamicForceModels(num_envs, device, debug=False)

    # ========== 测试1: 浮力 ==========
    print("\n===== 测试1: 浮力计算 (calculate_buoyancy_forces) =====")

    # 四元数 [w, x, y, z]
    root_quats = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],              # 无旋转
        [0.7071068, 0.7071068, 0.0, 0.0],   # 绕X轴90°
        [0.7071068, 0.0, 0.7071068, 0.0],   # 绕Y轴90°
        [0.7071068, 0.0, 0.0, 0.7071068],   # 绕Z轴90°
    ], device=device)

    F = volume * water_rho * g_mag  # 浮力大小
    T = 0.3 * F                     # 扭矩大小 (r=0.3)

    # 期望浮力 (本体系): quat_apply_inverse(q, [0,0,1]) * F
    expected_forces = torch.tensor([
        [0.0, 0.0, F],    # 无旋转: [0,0,1]
        [0.0, F, 0.0],    # X90°: R_x^T * [0,0,1] = [0,1,0]
        [-F, 0.0, 0.0],   # Y90°: R_y^T * [0,0,1] = [-1,0,0]
        [0.0, 0.0, F],    # Z90°: R_z^T * [0,0,1] = [0,0,1]
    ], device=device)

    # 期望扭矩 (本体系): r × F, 其中 r=[0,0,0.3]
    # [0,0,0.3] × [Fx,Fy,Fz] = [-0.3*Fy, 0.3*Fx, 0]
    expected_torques = torch.tensor([
        [0.0, 0.0, 0.0],    # r × [0,0,F] = [0,0,0]
        [-T, 0.0, 0.0],     # r × [0,F,0] = [-T,0,0]
        [0.0, -T, 0.0],     # r × [-F,0,0] = [0,-T,0]
        [0.0, 0.0, 0.0],    # r × [0,0,F] = [0,0,0]
    ], device=device)

    b_force, b_torque = model.calculate_buoyancy_forces(
        root_quats, water_rho, volume_tensor, g_mag, com_to_cob_offset
    )
    assert_close("浮力", b_force, expected_forces)
    assert_close("浮力扭矩", b_torque, expected_torques)

    # ========== 测试2: 零速度时阻力为零 ==========
    print("\n===== 测试2: 零速度阻力 (应为零) =====")

    zero_vel = torch.zeros((num_envs, 3), device=device)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)

    f_d, g_d, f_v, g_v = model.calculate_density_and_viscosity_forces(
        identity_quat, zero_vel, zero_vel, inertias, water_beta, water_rho, masses
    )
    zeros = torch.zeros((num_envs, 3), device=device)
    assert_close("二次阻力", f_d, zeros)
    assert_close("二次阻力扭矩", g_d, zeros)
    assert_close("粘性力", f_v, zeros)
    assert_close("粘性扭矩", g_v, zeros)

    # ========== 测试3: 阻力方向性 ==========
    print("\n===== 测试3: 阻力方向性 (力应与速度方向相反) =====")

    forward_vel_w = torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
    zero_angvel = torch.zeros((num_envs, 3), device=device)

    f_d, g_d, f_v, g_v = model.calculate_density_and_viscosity_forces(
        identity_quat, forward_vel_w, zero_angvel, inertias, water_beta, water_rho, masses
    )

    # 阻力 x 分量应为负（与正向速度相反）
    drag_x_negative = (f_d[:, 0] < 0).all().item()
    viscous_x_negative = (f_v[:, 0] < 0).all().item()
    # y, z 分量应为零（速度只有 x 分量）
    drag_yz_zero = torch.allclose(f_d[:, 1:], zeros[:, 1:], atol=1e-6)
    viscous_yz_zero = torch.allclose(f_v[:, 1:], zeros[:, 1:], atol=1e-6)

    if drag_x_negative and drag_yz_zero:
        print(f"  ✓ 二次阻力方向正确 (f_d_x={f_d[0, 0]:.4f} N)")
    else:
        print(f"  ✗ 二次阻力方向错误: {f_d[0]}")

    if viscous_x_negative and viscous_yz_zero:
        print(f"  ✓ 粘性力方向正确 (f_v_x={f_v[0, 0]:.6f} N)")
    else:
        print(f"  ✗ 粘性力方向错误: {f_v[0]}")

    # ========== 测试4: body→world→body 往返一致性 ==========
    print("\n===== 测试4: 坐标变换往返一致性 =====")

    test_vec = torch.tensor([[1.0, 2.0, 3.0]], device=device).repeat(num_envs, 1)
    for i, label in enumerate(["无旋转", "X90°", "Y90°", "Z90°"]):
        q = root_quats[i:i+1].repeat(num_envs, 1)
        # body → world → body 应该还原
        v_world = quat_apply(q, test_vec)
        v_body = quat_apply_inverse(q, v_world)
        ok = torch.allclose(v_body, test_vec, atol=1e-5)
        status = "✓" if ok else "✗"
        print(f"  {status} {label}: 往返误差 {(v_body - test_vec).abs().max().item():.2e}")

    print("\n===== 所有测试完成 =====")
