#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    TIME_SCALE_ARG = 0  # 不做时间衰减；需要时可设为一局的帧数量级

    REWARD_WEIGHT_DICT = {
        # === 目标与基础（稳定方向感） ===
        "tower_hp_point":     3.0,
        "hero_hp_point":      1.5,
        "gold_point":         0.02,
        "minion_push_depth":  0.25,
        "forward":            0.05,

        # === 击杀稀疏信号（战果） ===
        "kill_event":         2.0,
        "death_event":       -2.0,
        "kill_finish_bonus":  0.8,

        # === 安全/风险（抑制坏习惯） ===
        "tower_danger":      -0.45,
        "dive_no_minion":    -0.45,
        "low_hp_retreat":     0.30,
        "chase_low_enemy":    0.40,
        "grass_engage":       0.15,

        # === 即时对拼质量（密集、但幅度小） ===
        "hp_damage_adv":      0.002,

        # === 孙尚香技能/连招（核心） ===
        # 二技能（11120）——命中降甲 → 标记窗口内普攻加成
        "skill2_hit_w":       0.8,
        "aa_on_s2_mark":      0.6,

        # 一技能（11110）——翻滚位移 → 强化普攻
        "enhanced_aa_after_s1": 1.2,
        "s1_11110_use_chase":   0.4,
        "s1_11110_use_escape":  0.2,

        # 二→一→强普 三连（大奖，稀疏）
        "combo_2_1_aa":        1.8,

        # 三技能（11130）——远程补伤/收割
        "skill3_hit_w":        0.6,

        # === 可选项（按需开启，避免重复计分） ===
        # "ep_rate":           0.75,
        "exp_point":         0.006,
        "last_hit_event":    0.05,
        # "skill1_hit":        0.0,   # 若用 skill1_hit_w/强化普攻系，建议保持 0
        # "skill2_hit":        0.0,   # 与 skill2_hit_w 重复时保持 0
        # "combo_21":          0.0,   # 已有 combo_2_1_aa，更符合孙尚香连招
    }


    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_FEATURE = [260]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        260 + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(260,), (85,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(260 + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])

    # 自注意力参数
    USE_SELF_ATTENTION = True   # 开关
    SA_TOKENS = 4               # 虚拟token数T
    SA_DIM = 64                 # 每个token维度D，需满足 T * D == 256
    SA_HEADS = 4                # Multi-Head
    SA_LAYERS = 2               # Transformer Encoder 层数
    SA_DROPOUT = 0.0            # 注意力/FFN dropout

    # Dual_ppo
    USE_DUAL_CLIP_PPO = True
    DUAL_CLIP_C = 2.0
