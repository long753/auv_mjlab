"""CQU AUV velocity environment configurations (minimal runnable)."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactSensorCfg
from src.assets.robots import get_auv_robot_cfg
import mjlab.tasks.velocity.velocity_env_cfg as _base_velocity_env_cfg


def _make_base_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
    """
    从 mjlab 内置的速度环境配置模块中获取基础配置对象。
    依次尝试可能的工厂函数名称，找到第一个可调用的并返回其返回值。
    """
    for fn_name in (
        "make_velocity_env_cfg",
        "velocity_env_cfg",
        "get_velocity_env_cfg",
        "make_env_cfg",
    ):
        fn = getattr(_base_velocity_env_cfg, fn_name, None)
        if callable(fn):
            cfg = fn()
            if isinstance(cfg, ManagerBasedRlEnvCfg):
                return cfg
    raise RuntimeError(
        "Cannot find base velocity env factory in mjlab.tasks.velocity.velocity_env_cfg."
    )


def cqu_auv_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """
    为 CQU AUV 配置一个平面地形下的速度跟踪环境。

    Args:
        play: 若为 True，则进入演示模式（无 curriculum、超长 episode、禁用观测噪声）。

    Returns:
        配置好的 ManagerBasedRlEnvCfg 对象。
    """
    # 获取基础的速度环境配置
    cfg = _make_base_velocity_env_cfg()

    # ------------------- 替换机器人为 AUV -------------------
    cfg.scene.entities = {"robot": get_auv_robot_cfg()}

    # ------------------- 使用平面地形 -------------------
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    # 移除 curriculum 中的地形级别设置（因为平面地形无需分级）
    if cfg.curriculum is not None:
        cfg.curriculum.pop("terrain_levels", None)

    # ------------------- 移除相关的传感器 -------------------
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if getattr(s, "name", "") != "terrain_scan"
        and not isinstance(s, ContactSensorCfg)
    )

    # ------------------- 观测空间 -------------------
    # 仅保留指令（command）和动作（actions），删除其他观测项
    keep_obs = {"command", "actions"}
    for group in ("actor", "critic"):
        if group in cfg.observations:
            terms = cfg.observations[group].terms
            cfg.observations[group].terms = {k: v for k, v in terms.items() if k in keep_obs}

    # ------------------- 动作空间 -------------------
    if "joint_pos" in cfg.actions:
        act = cfg.actions["joint_pos"]
        if hasattr(act, "actuator_names"):
            # 匹配 XML 中定义的关节名称（joint_servo_up/down/left/right）
            act.actuator_names = ("joint_servo_.*",)
        if hasattr(act, "scale"):
            act.scale = 1.0
        if hasattr(act, "use_default_offset"):
            # 使关节零位对应于默认姿态（即推进器初始位置）
            act.use_default_offset = True

    # ------------------- 奖励函数 -------------------
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards = {"action_rate_l2": cfg.rewards["action_rate_l2"]}
        cfg.rewards["action_rate_l2"].weight = -0.01
    else:
        cfg.rewards = {}

    # ------------------- 终止条件 -------------------
    if "time_out" in cfg.terminations:
        cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    else:
        cfg.terminations = {}

    # ------------------- 调整可视化参数 -------------------
    cfg.viewer.body_name = "base"          # 跟随基座
    if hasattr(cfg.viewer, "distance"):
        cfg.viewer.distance = 2.5          # 相机距离
    if hasattr(cfg.viewer, "elevation"):
        cfg.viewer.elevation = -15.0       # 俯视角度

    # ------------------- 调整仿真参数 -------------------
    # 增加关节和接触数量上限（AUV 模型可能有较多关节/接触）
    if hasattr(cfg.sim, "njmax"):
        cfg.sim.njmax = 2048
    if hasattr(cfg.sim, "nconmax"):
        cfg.sim.nconmax = 1024
    if hasattr(cfg.sim, "contact_sensor_maxmatch"):
        cfg.sim.contact_sensor_maxmatch = 128

    # ------------------- 演示模式特殊处理 -------------------
    if play:
        cfg.episode_length_s = int(1e9)          # 极长 episode
        if "actor" in cfg.observations:
            cfg.observations["actor"].enable_corruption = False  # 关闭观测噪声
        cfg.curriculum = {}                      # 清空课程学习

    # ------------------- 移除不适用的随机事件 -------------------
    cfg.events.pop("push_robot", None)      # 水下不需要推力扰动
    cfg.events.pop("reset_base", None)      # 无需重置基座

    # 重置关节事件：将四个推进器关节初始位置/速度强制设为 0
    if "reset_robot_joints" in cfg.events:
        p = cfg.events["reset_robot_joints"].params
        if "asset_cfg" in p and hasattr(p["asset_cfg"], "joint_names"):
            p["asset_cfg"].joint_names = ("joint_servo_.*",)
        p["position_range"] = (0.0, 0.0)
        p["velocity_range"] = (0.0, 0.0)

    return cfg