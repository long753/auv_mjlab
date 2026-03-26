## 如何利用VNC远程桌面训练

### 远程端操作

1.首先在vscode进入远程ssh

```bash
ssh robot1@10.253.33.188
```

- 注意一般需要先连接到同一个局域网，并且确保进入的时候关闭魔法，进入之后貌似可以打开

2.然后打开远程终端，这一步在vscode里面完成即可。最好是在base环境下

```bash
vncserver -kill :1
```

```bash
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
```



### 本地操作

在本地打开远程桌面
```bash
vncviewer 10.253.33.188:1
```

密码：147258




## 参考架构
@ 宇树科技
https://github.com/unitreerobotics/unitree_rl_mjlab



## 训练方法
#### 训练
```
python scripts/train.py Mjlab-Velocity-Flat-CQU-AUV --env.scene.num-envs=并行数量 --agent.max-iterations=回合数量
```
#### 远程回放
```
python scripts/play.py Mjlab-Velocity-Flat-CQU-AUV   --checkpoint-file logs/rsl_rl/auv_velocity/时间戳/model_第几个回合.pt   --num-envs 数字   --viewer viser   --no-terminations True
```
- 会返回一个网址，链接即可