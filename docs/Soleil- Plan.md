正常离线渲染
首先需要看看正常情况它是怎么渲染的，我们先专攻heading
会生成scripts/results/singlecontrol/1/heading/ppo/v1/render/v1.txt.acmi文件件,[tacview acmi官方文档l(htps://www.tacview. net/16documentation/acmi/en/)使用tacview打开acmi文件后，使用ffmpeg生成gif图片
# 查找文件：command+P
# 进入真正的项目根目录:cd CloseAirCombat-master

1.JSBSim环境
# 解释器选择 （command+shift+p）：
jsbsim
# 退出当前的 venv 虚拟环境
deactivate
# 激活我们真正的主力环境
conda activate jsbsim

2.test_env.py
-跑通环境，（2v2/NoWeapon/HierarchySelfplay）（1/heading）
    飞机模型：f16
    输出文件JSBSimRecording.txt.acmi，JSBSimRecording_SingleControl.txt.acmi
-windows的Tacview中渲染

3.Scripts-启动器
-超参数（如学习率、PPO 截断范围、环境并行数量、使用的 GPU 等）
    train_heading.sh、train_selfplay.sh、 train_share_selfplay.sh
    核心运行：scripts/train/train_jsbsim
-在启动之前，系统会实例化两个关键的神经网络：
    Actor (策略网络)：看一眼仪表盘（State），决定推多少油门、拉多少杆（Action）。
    Critic (价值网络)：看一眼全局局势（Share State），给当前状态打个分（Value），告诉 Actor 刚才那步走得好不好。
-human_combat
-render
-train
-.sh(cd scripts)
# bash train_heading.sh
    -train_heading.sh(参数包装)
        env="SingleControl" scenario="1/heading"
        并行线程数（n-rollout-threads）、学习率（lr）、截断范围（clip-params）
        stop:100000000
        输出：reward return
    - train_selfplay.sh
-results/SingleControl/1/heading/ppo/v1



4.envs
-JSBSim
 -configs：配置参数（比如初始高度、速度、经纬度）
 -core：将PPO——actor中 Python 里的数字精准映射到 JSBSim 物理引擎底层的控制面节点

5.algorithms
-ppo
 -ppo_actor:Actor 神经网络。它看到仪表盘和雷达数据后，每一步直接输出一个包含 4 个数字的数组，飞行员推拉操纵杆的力度：
 【副翼（滚转）、升降舵（俯仰）、方向舵（偏航）和油门】。

6.runner-训练引擎

7.renders
-render_1v1
    两个已经练成“绝世武功”的 AI 大脑装进飞机
    自我对弈 (Self-Play)ego_policy_index = 1040.  enm_policy_index = 0
    实战状态 输出当前认知下胜率最高的、最确定的最优动作

-render_2v2_indepent
-render_2v3