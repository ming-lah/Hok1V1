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
        
        # === 新增：孙尚香专用增强奖励 ===
        # 基于论文洞察的分层奖励设计
        "perfect_combo_s2_s1_aa":    5.0,   # 完美三连
        "kiting_execution":          2.0,   # 风筝战术执行
        "positioning_quality":       1.8,   # 位置质量
        "safe_farming":              1.2,   # 安全发育
        "tactical_retreat":          0.8,   # 战术撤退
        "aggressive_pursuit":        1.5,   # 追击质量
        "skill_combo_timing":        1.0,   # 技能时机
        "marksman_spacing":          1.3,   # 射手走位
        
        # === 新增：经济博弈奖励系统 ===
        # 基于零和博弈思路的经济决策奖励
        "economic_hero_priority":    2.5,   # 经济优势时优先攻击英雄
        "economic_farming_priority": 1.8,   # 经济优势时转向发育
        "economic_catchup_farming":  2.2,   # 经济劣势时追赶发育
        "economic_safe_farming":     1.5,   # 安全发育
        "economic_pursuit":          0.8,   # 经济优势时的追击
        "economic_efficiency":       0.5,   # 经济效率
        "economic_equipment_timing": 0.6,   # 装备购买时机
        "economic_minion_penalty":  -0.5,   # 错失攻击英雄机会的惩罚
        
        # 经济压制与防守
        "economic_pressure":         1.5,   # 经济压制
        "economic_defense":          1.3,   # 经济防守
        "economic_comeback":         2.0,   # 经济翻盘
        "last_hit_accuracy":         1.0,   # 补刀准确性
        "farming_speed":             0.8,   # 发育速度
        "economic_growth_rate":      1.2,   # 经济增长率
        
        # === 新增：稠密奖励系统权重 ===
        # 基于深度RL分析的完整奖励塑形
        
        # 核心目标奖励
        "game_victory":              1000.0,  # 赢得比赛
        "game_defeat":              -1000.0,  # 输掉比赛
        "tower_destroy":             1000.0,  # 摧毁敌方防御塔
        "tower_destroyed":          -1000.0,  # 我方防御塔被摧毁
        
        # 推塔相关奖励
        "tower_damage_dealt":         2.0,    # 对防御塔造成伤害
        "tower_damage_received":     -2.0,    # 我方防御塔受到伤害
        "tower_hp_advantage":         5.0,    # 防御塔血量优势维持
        
        # 英雄对战奖励
        "hero_damage_dealt":          1.0,    # 对敌方英雄造成伤害
        "hero_damage_received":      -1.0,    # 受到敌方英雄伤害
        "hero_kill":                200.0,    # 击杀敌方英雄
        "hero_death":               -200.0,   # 自己死亡
        "hp_advantage_maintain":      1.0,    # 维持血量优势
        
        # 经济发育奖励 (与现有经济奖励协调)
        "gold_gain":                  0.02,   # 金币获取
        "exp_gain":                   0.01,   # 经验获取
        "last_hit_success":           5.0,    # 成功补刀
        "last_hit_miss":             -2.0,    # 错失补刀
        "farming_efficiency":         3.0,    # 发育效率
        
        # 生存与血量管理
        "hp_ratio_maintain":          1.0,    # 维持健康血量
        "low_hp_penalty":            -2.0,    # 低血量惩罚
        "hp_recovery":                2.0,    # 血量回复奖励
        "safe_positioning":           1.5,    # 安全位置奖励
        
        # 技能使用奖励
        "skill_hit_hero":             3.0,    # 技能命中英雄
        "skill_miss":                -1.0,    # 技能未命中
        "combo_execution":           10.0,    # 连招执行
        "skill_cd_management":        1.0,    # 技能冷却管理
        
        # 位置与移动奖励
        "position_advantage":         2.0,    # 位置优势
        "safe_retreat":               3.0,    # 安全撤退
        "aggressive_advance":         2.0,    # 主动进攻
        "movement_efficiency":        1.0,    # 移动效率
        
        # 战术执行奖励
        "tempo_control":              2.0,    # 节奏控制
        "resource_control":           2.0,    # 资源控制
        
        # 时序奖励
        "early_game_farming":         2.0,    # 前期发育
        "mid_game_fighting":          3.0,    # 中期对战
        "late_game_finishing":        4.0,    # 后期终结
        "game_pace_control":          1.5,    # 游戏节奏控制
        
        # 惩罚机制
        "inefficient_action":        -1.0,    # 无效行动
        "resource_waste":            -2.0,    # 资源浪费
        "tactical_error":            -3.0,    # 战术错误
        "positioning_error":         -2.0,    # 位置错误
        
        # === 新增：防御塔战略奖励系统权重 ===
        # 基于防御塔作为1v1唯一胜利条件的深度分析
        
        # 终端奖励 (Terminal Rewards) - 针对拆塔胜利条件大幅提升
        "tower_victory":             2000.0,  # 摧毁敌方防御塔 (胜利) - 提升到2000
        "tower_defeat":             -2000.0,  # 我方防御塔被摧毁 (失败) - 提升到-2000
        
        # 事件驱动型奖励 (Event-Driven Rewards) - 提升推塔相关奖励
        "enemy_tower_damage":         8.0,    # 对敌方防御塔造成伤害 - 从5.0提升到8.0
        "my_tower_damage":           -8.0,    # 我方防御塔受到伤害 - 从-5.0提升到-8.0
        "minion_tower_damage":        3.0,    # 我方小兵对敌方塔造成伤害 - 从2.0提升到3.0
        "tower_shot_penalty":       -25.0,    # 被敌方防御塔攻击的严重惩罚 - 从-20.0提升到-25.0
        
        # 基于势能的奖励 (Potential-Based Rewards)
        "tower_hp_potential":       100.0,    # 防御塔血量差势能权重
        
        # 兵线运营奖励 (Wave Management Rewards)
        "wave_crash_bonus":          50.0,    # 送兵进塔奖励
        "perfect_clear_bonus":       30.0,    # 完美解线奖励
        "wave_control_bonus":        20.0,    # 兵线控制奖励
        
        # 智能推塔奖励 (Smart Pushing Rewards)
        "smart_tower_push":          10.0,    # 有兵线优势时推塔的额外奖励
        "blind_push_penalty":       -15.0,    # 无兵线强推的惩罚
        
        # 动态发育奖励 (Dynamic Farming Rewards)
        "pressure_farming_penalty": -10.0,   # 防守压力下贪图发育的惩罚
        "safe_farming_bonus":         5.0,   # 安全发育奖励
        
        # === 新增：胜利条件导向奖励系统权重 ===
        # 专门针对拆塔胜利条件的高级奖励机制
        
        # 胜利导向奖励 (Victory-Focused Rewards)
        "endgame_urgency_bonus":     50.0,    # 终局紧迫状态下的行动奖励
        "one_push_victory_bonus":   100.0,    # 一波结束游戏的奖励
        "desperate_defense_bonus":   80.0,    # 绝境防守成功的奖励
        "tower_hp_amplified_bonus": 200.0,    # 终局阶段塔血量优势的放大奖励
        "decisive_timing_bonus":     60.0,    # 把握决战时机的奖励
        "critical_tower_save":      150.0,    # 拯救危险防御塔的奖励
        "finishing_blow_bonus":     300.0,    # 最后一击摧毁敌方塔的额外奖励
        
        # 战略失误惩罚 (Strategic Blunder Penalties)
        "greedy_death_penalty":     -500.0,   # 贪婪致死的严重惩罚
        "missed_opportunity_penalty": -100.0, # 错失推塔机会的惩罚
        "poor_timing_penalty":       -50.0,    # 时机选择不当的惩罚
        
        # === 新增：兵线战略奖励系统权重 ===
        # 基于兵线作为MOBA游戏"血液"的深度分析
        
        # 经济收益奖励 (Economic Gain Rewards)
        "last_hit_gold":             1.0,     # 成功补刀金币奖励
        "last_hit_exp":              1.5,     # 成功补刀经验奖励 (前期更重要)
        "miss_last_hit_penalty":    -0.5,     # 漏刀惩罚
        "perfect_last_hit_bonus":     2.0,     # 完美补刀奖励 (炮车等)
        
        # 兵线控制奖励 (Wave Control Rewards) - 势能奖励
        "wave_advantage_potential":  50.0,    # 兵线优势势能权重
        "freeze_sustain_bonus":       2.0,    # 稳定控线持续奖励 (每秒)
        "exp_deny_bonus":             5.0,    # 经验剥夺事件奖励
        "super_wave_formation":     100.0,    # 超级兵线形成奖励
        "wave_crash_bonus":          50.0,    # 兵线撞塔奖励
        "perfect_recall_bonus":      30.0,    # 完美回城奖励
        
        # 战略交互奖励 (Strategic Interaction Rewards)
        "minion_aggro_penalty":     -10.0,    # 顶兵线战斗惩罚
        "minion_shield_efficiency":   1.5,    # 利用小兵做盾牌的效率奖励
        "wave_timing_mastery":       20.0,    # 兵线时机掌控奖励
        
        # 高级战略奖励 (Advanced Strategic Rewards)
        "slow_push_execution":       40.0,    # 慢推执行奖励
        "fast_push_reset":           25.0,    # 快推重置奖励
        "freeze_break_timing":       15.0,    # 破解控线时机奖励
        "lane_state_transition":     10.0,    # 兵线状态转换奖励
        
        # 兵线惩罚机制 (Minion Penalty Mechanisms)
        "poor_wave_management":     -20.0,    # 兵线管理不当惩罚
        "missed_farming_window":    -15.0,    # 错失发育窗口惩罚
        "inefficient_recall":       -25.0,    # 低效回城惩罚
        "strategic_blunder":        -30.0,    # 战略失误惩罚
        
        # === 新增：草丛战术奖励系统权重 ===
        # 基于草丛作为1v1对线中最重要战术资源的深度分析
        
        # 成功伏击奖励 (Successful Ambush Rewards)
        "first_strike_bonus":        15.0,    # 先手优势奖励 (额外奖励，基础伤害奖励*倍数)
        "free_poke_bonus":           50.0,    # 无伤消耗奖励
        "perfect_ambush_bonus":     100.0,    # 完美伏击奖励
        "stealth_attack_bonus":      30.0,    # 隐身攻击奖励
        
        # 信息控制与心理压迫奖励 (Info Control & Pressure)
        "vision_control_continuous":  5.0,    # 视野压制持续奖励 (每秒)
        "information_asymmetry_bonus": 15.0,   # 信息不对称奖励
        "psychological_pressure":     20.0,    # 心理压迫奖励
        "enemy_hesitation_reward":    25.0,    # 敌方犹豫奖励
        
        # 战术性规避与重置奖励 (Tactical Evasion & Reset)
        "aggro_reset_bonus":         40.0,    # 仇恨重置奖励
        "skill_dodge_bonus":         80.0,    # 躲避关键技能奖励
        "tower_aggro_reset":         60.0,    # 防御塔仇恨重置奖励
        "strategic_retreat_bonus":   35.0,    # 战略撤退奖励
        
        # 孙尚香特色连招奖励 (Sun Shangxiang Special Combo)
        "sunshangxiang_bush_combo": 120.0,    # 孙尚香草丛连招奖励
        "enhanced_auto_from_bush":   60.0,    # 草丛强化普攻奖励
        "roll_bush_escape":          45.0,    # 翻滚进草逃脱奖励
        "bush_kiting_mastery":       70.0,    # 草丛风筝精通奖励
        
        # 高级草丛战术奖励 (Advanced Bush Tactics)
        "bush_control_dominance":    50.0,    # 草丛控制统治奖励
        "vision_game_mastery":       40.0,    # 视野博弈精通奖励
        "timing_perfection":         35.0,    # 时机完美奖励
        "bush_mind_games":           45.0,    # 草丛心理博弈奖励
        
        # 草丛惩罚机制 (Bush Penalty Mechanisms)
        "blind_bush_entry":         -20.0,    # 盲目进草惩罚
        "poor_bush_timing":         -25.0,    # 草丛时机不当惩罚
        "wasted_stealth_opportunity": -30.0,  # 浪费隐身机会惩罚
        "exposed_positioning":      -15.0,    # 暴露位置惩罚

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
    DIM_OF_FEATURE = [217]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        217 + 85,
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
    SERI_VEC_SPLIT_SHAPE = [(217,), (85,)]
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
        [(217 + 85) * 16],
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

    # Dual_ppo - 基于论文2的双剪切PPO优化
    USE_DUAL_CLIP_PPO = True
    DUAL_CLIP_C = 2.0
    CLIP_PARAM = 0.2
    
    # === 新增：基于论文洞察的增强配置 ===
    # 目标注意力机制配置
    USE_TARGET_ATTENTION = True
    TARGET_ATTENTION_DIM = 128
    NUM_ACTION_TARGETS = 9
    
    # 孙尚香专用连招检测配置
    USE_COMBO_DETECTOR = True
    COMBO_DETECTOR_DIM = 64
    COMBO_LSTM_LAYERS = 1
    
    # 增强特征处理配置
    USE_ENHANCED_FEATURES = True
    ENHANCED_FEATURE_DIM = 100  # 孙尚香专用增强特征维度
    TEMPORAL_FEATURE_WINDOW = 16  # 时序特征窗口长度
    
    # 自适应奖励权重配置
    USE_ADAPTIVE_REWARDS = True
    REWARD_ADAPTATION_FREQUENCY = 100
    
    # 课程学习配置
    USE_CURRICULUM_LEARNING = True
    CURRICULUM_PHASES = {
        'early': 10000,    # 前10k步：基础操作
        'mid': 50000,      # 10k-50k步：连招训练  
        'late': float('inf')  # 50k+步：全面训练
    }
