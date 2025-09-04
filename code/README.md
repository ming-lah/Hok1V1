# 王者荣耀1v1强化学习项目

## 项目结构

```
code/
├── agent_ppo/                    # PPO智能体实现
│   ├── agent.py                  # 主要智能体类
│   ├── algorithm/                # PPO算法实现
│   ├── conf/                     # 配置文件
│   ├── feature/                  # 特征处理和奖励系统
│   │   ├── feature_process/      # 特征处理模块
│   │   ├── *_processor.py        # 各种战略特征处理器
│   │   ├── *_rewards.py          # 各种奖励系统
│   │   └── reward_process.py     # 主要奖励处理
│   ├── model/                    # 神经网络模型
│   └── workflow/                 # 训练工作流
├── conf/                         # 全局配置文件
├── kaiwu.json                    # KaiWu配置
└── train_test.py                 # 训练启动脚本
```

## 核心功能

- **多维度特征工程**: 220维完整特征 (全面特征125维 + 防御塔30维 + 兵线40维 + 草丛25维)
- **综合奖励系统**: 76+项奖励机制，涵盖经济博弈、防御塔战略、兵线运营、草丛战术
- **孙尚香专用优化**: 针对孙尚香英雄的特殊技能和连招优化
- **战略AI思维**: 从反应式到规划式的AI决策升级

## 使用方法

1. 配置环境和依赖
2. 运行 `python train_test.py` 开始训练
3. 算法配置在 `agent_ppo/conf/conf.py` 中调整奖励权重
