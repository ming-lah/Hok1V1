#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

改动要点（保守策略）：
- 仅在“塔相关”加入小额每帧惩罚项（负权重，幅度很小）；
- forward_progress 降权并加安全门（在 reward_process 中实现）；
- 引入势函数开关与强度；其余算法参数不动（如需“重热探索”，再行调整）。
"""

class GameConfig:
    # 统一只保留一处定义，先关闭时间衰减（避免与折扣 GAMMA 叠加）
    TIME_SCALE_ARG = 0

    # ——奖励权重（保守调整）——
    REWARD_WEIGHT_DICT = {
        # 原有项
        "hp_point":            2.0,
        "tower_hp_point":      8.0,     # ↑ 强化推塔目标
        "money":               0.0035,  # ↓ 与 last_hit 避免双计分
        "ep_rate":             0.40,    # ↓
        "death":              -1.60,    # |death| > kill，抑制 1 换 1
        "kill":                1.20,    # ↓
        "exp":                 0.0035,  # ↓
        "last_hit":            0.25,    # ↓
        "forward_progress":    0.0003,  # ↓（安全门控制；若关势函数可再微调）

        # 新增（保守的每帧小额惩罚；只有在敌塔相关触发）
        "under_enemy_tower":  -0.020,   # 在敌塔攻击圈内
        "tower_aggro":        -0.080,   # 被敌塔锁定/集火
        "low_hp_under_tower": -0.050,   # 塔下且血量低于阈值
    }

    # frame_action 一维事件向量（用于 kill/death/last_hit）
    FRAME_ACTION_VECTOR_MAP = {
        "my_kill_hero": 0,
        "my_death": 1,
        "my_last_hit_soldier": 2,
    }
    FRAME_ACTION_VECTOR_IS_CUMULATIVE = True
    FRAME_ACTION_VECTOR_IS_BINARY = False

    # 前进奖励裁剪设置
    FORWARD_PROGRESS_CLIP_NEGATIVE = True
    FORWARD_PROGRESS_STEP_CAP = 60.0

    # ——安全门阈值——
    SAFE_HP_RATE = 0.55            # 塔下血量低于此阈值强惩罚
    MINION_ADV_TO_DIVE = 2         # 塔圈内（我方兵-敌方兵）需要 ≥ 2 才允许前进得分
    TOWER_RANGE_MARGIN = 1.15      # 敌塔攻击半径安全裕度
    MAP_RADIUS_FOR_PROGRESS = 2500.0  # 用于推进度归一的地图尺度

    # 势函数：不改变最优策略，可一键关闭
    USE_POTENTIAL_SHAPING = True
    PHI_BETA = 0.15

    # 模型保存间隔
    MODEL_SAVE_INTERVAL = 1800


# 维度配置
class DimConfig:
    DIM_OF_FEATURE = [310]


# 模型与算法配置（保持你的原值，必要时再单独调参）
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        310 + 85,
        1, 1, 1, 1, 1, 1, 1, 1,
        12, 16, 16, 16, 16, 9,
        1, 1, 1, 1, 1, 1, 1,
        LSTM_UNIT_SIZE, LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(310,), (85,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(310 + 85) * 16],
        [16], [16], [16], [16], [16], [16], [16], [16],
        [192], [256], [256], [256], [256], [144],
        [16], [16], [16], [16], [16], [16], [16],
        [512], [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])

    # 自注意力
    USE_SELF_ATTENTION = True
    SA_TOKENS = 4
    SA_DIM = 64
    SA_HEADS = 4
    SA_LAYERS = 2
    SA_DROPOUT = 0.0  # 如需抑制早熟，可改为 0.1

    # 时序记忆骨干：'lstm' 或 'mem_transformer'
    MEMORY_BACKBONE = 'lstm'
    # LSTM 记忆开关（当 MEMORY_BACKBONE == 'lstm' 有效）
    USE_LSTM_MEMORY = True
    # 记忆型 Transformer 相关超参（当 MEMORY_BACKBONE == 'mem_transformer' 有效）
    MEM_TOKENS = 2          # 每步注入的记忆 token 数
    MEM_DROPOUT = 0.0       # 编码层内的 dropout（沿用 SA_DIM/SA_HEADS/SA_LAYERS）

    # Dual-Clip PPO
    USE_DUAL_CLIP_PPO = True
    DUAL_CLIP_C = 2.0
