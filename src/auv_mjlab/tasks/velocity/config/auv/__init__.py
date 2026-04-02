from mjlab.tasks.registry import register_mjlab_task
from src.auv_mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfgs import cqu_auv_flat_env_cfg
from .rl_cfg import cqu_auv_ppo_runner_cfg
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-CQU-AUV",
  env_cfg=cqu_auv_flat_env_cfg(),
  play_env_cfg=cqu_auv_flat_env_cfg(play=True),
  rl_cfg=cqu_auv_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)