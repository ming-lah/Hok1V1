# 🏆 王者荣耀1v1强化学习AI - 完整特征工程文档

## 📊 特征体系总览

本项目实现了**630维**的综合特征体系，涵盖英雄、环境、战术、心理等多个维度，旨在训练出具备职业选手水平的1v1对战AI。

### 🎯 特征分布概况

| 特征模块 | 维度数 | 核心功能 | 战术价值 |
|----------|--------|----------|----------|
| **英雄核心特征** | 132维 | 技能预测、战斗距离、战术预判 | ⭐⭐⭐⭐⭐ |
| **基础环境特征** | 148维 | 塔、经济、游戏阶段、基础兵线 | ⭐⭐⭐⭐ |
| **野怪资源特征** | 40维 | 野怪控制、机会成本、时机把握 | ⭐⭐⭐⭐ |
| **高级防御塔特征** | 50维 | 动态风险、战略交换、心理博弈 | ⭐⭐⭐⭐⭐ |
| **专家级兵线特征** | 60维 | 控线、慢推、回推、虚实博弈 | ⭐⭐⭐⭐⭐ |
| **辅助处理特征** | 200维 | 基础兵线处理（已优化） | ⭐⭐⭐ |

**总计：630维特征**

---

## 🎖️ 核心特征模块详解

### 1. 🦸 英雄核心特征模块 (132维)
*实现位置: `agent_ppo/feature/feature_process/hero_process.py`*

#### 1.1 技能预测与连招特征 (12维)
让AI具备预判敌方行为和执行完美连招的能力。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `skill_combo_ready` | 检查多技能组合可用性 | 连招时机判断 |
| `skill_1_enemy_in_range` | 一技能是否能命中敌方 | 技能释放决策 |
| `displacement_skill_available` | 位移技能可用性 | 进攻/逃脱时机 |
| `ultimate_combo_ready` | 大招连招就绪状态 | 关键团战时机 |
| `enemy_skill_threat_level` | 敌方技能威胁评估 | 防御姿态调整 |
| `dodge_window_available` | 躲避技能窗口期 | 微操时机把握 |

#### 1.2 战斗距离与位置特征 (10维)
精确的距离控制和位置优势计算。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `optimal_fight_distance` | 基于英雄类型的最优交战距离 | 战斗姿态选择 |
| `kite_distance_advantage` | 走砍距离优势评估 | 风筝战术执行 |
| `escape_route_available` | 逃跑路径可用性 | 生存决策支持 |
| `terrain_advantage` | 地形优势评估 | 位置战术制定 |
| `flanking_opportunity` | 侧翼包抄机会 | 进攻路线选择 |

#### 1.3 高级战术预测特征 (8维)
基于AI深度学习的行为预测和战术分析。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `enemy_next_action_predict` | 基于历史行为的下一步预测 | 先发制人 |
| `burst_combo_window` | 爆发连招机会窗口 | 秒杀时机把握 |
| `zone_control_advantage` | 区域控制优势 | 地图控制力 |
| `prediction_confidence` | 预测结果置信度 | 决策风险评估 |

---

### 2. 🏰 高级防御塔特征模块 (50维)
*实现位置: `agent_ppo/feature/feature_process/organ_process.py` - `_encode_advanced_tower_features`*

这是将AI从"操作手"提升为"战术大师"的关键模块。

#### 2.1 动态攻防风险评估特征 (15维)
精确量化每一次塔下操作的风险与收益。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `my_tower_time_to_live_under_push` | `塔血量 / 敌方总DPS` | 回防紧迫性评估 |
| `enemy_hero_dive_threat_score` | `敌方血量比 × (1+小兵数) / (我方血量比+0.1)` | 越塔威胁量化 |
| `hero_tower_tanking_endurance_sec` | `英雄血量 / (塔伤害×(1-护甲减免))` | 扛塔能力计算 |
| `tower_escape_time_sec` | `逃离距离 / 英雄移速` | 逃脱时间预估 |
| `hero_dive_survivability_margin` | `扛塔时间 - 逃脱时间` | **核心生存空间** |
| `ally_dive_execution_window` | 综合血量、小兵、敌方威胁评估 | 越塔执行时机 |
| `tower_teamfight_win_probability` | 塔下团战胜率预测 | 是否应该接团 |

#### 2.2 战略机会与交换评估特征 (15维)
让AI学会"算计"，评估每次资源交换的价值。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `tower_damage_potential_per_wave` | `小兵DPS × 存活时间` | 兵线推塔价值 |
| `enemy_tower_hp_delta_last_10s` | 历史血量变化追踪 | 攻势成效评估 |
| `objective_trade_advantage_score` | 推塔vs打野的价值对比 | 目标优先级决策 |
| `tower_resource_investment_roi` | `战略价值 / (时间成本+血量成本)` | 投入产出比分析 |
| `tower_offense_defense_transition_timing` | 攻防转换最佳时机 | 战略节奏控制 |
| `tower_map_control_advantage` | 塔周围地图控制力 | 区域优势评估 |

#### 2.3 微操时机与执行特征 (10维)
专注于精确的时机把握和微操执行。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `tower_attack_cooldown_exploitation` | 塔攻击间隔利用 | 极限微操时机 |
| `aggro_transfer_timing` | 仇恨转移最佳时机 | 塔下操作安全性 |
| `skill_cast_window_optimization` | 技能释放窗口优化 | 技能使用时机 |
| `kiting_rhythm_optimization` | 走砍节奏优化 | 完美风筝战术 |
| `tower_last_hit_timing` | 塔下补刀时机 | 经济收益最大化 |
| `escape_path_pre_planning` | 8方向逃跑路径评估 | 生存率提升 |

#### 2.4 心理博弈与压制特征 (10维)
模拟心理层面的博弈和压制效果。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `tower_suppression_effect` | 塔的心理威慑范围 | 心理压制效果 |
| `hp_psychological_advantage` | 血量差距的心理影响 | 心理优势评估 |
| `equipment_intimidation_factor` | 装备差距威慑效果 | 装备优势威慑 |
| `initiative_control_perception` | 主动权控制感知 | 节奏主导权 |
| `time_pressure_perception` | 游戏时长的心理压力 | 时间管理策略 |
| `comeback_potential_assessment` | 逆转潜力评估 | 翻盘机会把握 |

---

### 3. 🌊 专家级兵线特征模块 (60维)
*实现位置: `agent_ppo/feature/feature_process/organ_process.py` - `_encode_expert_wave_features`*

将AI从"清兵机器"升级为"兵线运营专家"。

#### 3.1 兵线健康度与构成细化特征 (15维)
深入分析兵线内部结构和健康状况。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `enemy_wave_one_shot_killable_count` | `血量 <= 英雄攻击力` 的小兵数 | 精确补刀预测 |
| `enemy_wave_one_skill_killable_count` | `血量 <= 技能伤害` 的小兵数 | 技能清兵决策 |
| `enemy_wave_hp_[low/medium/high]` | 血量在(0-33%/33-66%/66-100%)的小兵数 | 兵线健康度画像 |
| `wave_ranged_minion_advantage` | `我方远程兵数 - 敌方远程兵数` | **控线核心指标** |
| `ally/enemy_cannon_hp_ratio` | 炮车血量比例 | 推塔关键资源 |
| `aoe_clear_efficiency` | 基于兵线聚集度的AOE效率 | 清兵策略优化 |

#### 3.2 高级兵线控制与预测特征 (15维)
实现控线、慢推、回推等高级战术。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `wave_freeze_potential_score` | `距离因子×0.5 + 兵力差×0.3 + 远程兵优势×0.2` | **控线机会评估** |
| `wave_bounce_back_timer` | 兵线回推时间追踪 | **回推利用窗口** |
| `next_wave_arrival_time_ratio` | 下波兵线到达时间比例 | 时机规划支持 |
| `stacked_wave_potential_score` | `当前兵数 - 标准波数` | 屯兵线威力评估 |
| `hero_threat_to_freeze` | 敌方英雄对控线的威胁 | 控线风险评估 |
| `slow_push_identification` | 轻微优势+远程兵优势识别 | **慢推战术识别** |
| `wave_reset_timing` | 兵线重置需求评估 | 兵线管理策略 |

#### 3.3 兵线时序与节奏控制特征 (15维)
专注于兵线的时间维度和节奏掌控。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `wave_lifecycle_stage` | 基于血量分布的生命周期阶段 | 兵线状态感知 |
| `wave_breathing_rhythm` | 正弦波模拟推进-回撤循环 | 自然节奏掌握 |
| `last_hit_window_rhythm` | 即将进入补刀窗口的小兵预测 | 补刀节奏优化 |
| `wave_convergence_timing` | 新旧兵线汇合时机 | 兵线聚合利用 |
| `wave_energy_buildup` | `基础数量 + 炮车×2 + 远程兵×1.5` | 兵线战斗力评估 |
| `wave_roi_timing` | 不同游戏阶段的兵线价值 | 投入产出优化 |
| `wave_impact_prediction` | `总伤害 × 总血量` 的冲击力 | 兵线威胁评估 |

#### 3.4 兵线心理与战术博弈特征 (15维)
模拟兵线运营中的心理博弈和战术欺骗。

| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `wave_intimidation_effect` | 基于规模和构成的威慑效应 | 心理压制战术 |
| `wave_deception_potential` | 虚实博弈和战术欺骗潜力 | 高级战术运用 |
| `wave_psychological_advantage` | `数量优势×0.6 + 质量优势×0.4` | 兵线心理优势 |
| `wave_manipulation_difficulty` | 精确控制兵线的技术难度 | 操作复杂度评估 |
| `wave_rhythm_mastery` | 英雄对兵线节奏的控制程度 | 节奏掌控能力 |
| `wave_information_warfare` | 通过兵线进行信息战的价值 | 视野和情报获取 |
| `wave_endgame_thinking` | 游戏后期兵线决策的关键性 | 终局战略思维 |

---

### 4. 🎯 野怪资源特征模块 (40维)
*实现位置: `agent_ppo/feature/feature_process/organ_process.py` - `_encode_jungle_monster_features`*

精确评估野怪资源的争夺时机和机会成本。

#### 4.1 野怪自身与时序特征 (8维)
| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `monster_is_alive` | 野怪存在状态 | 资源可用性 |
| `monster_hp_ratio` | 当前血量比例 | 争夺难度评估 |
| `monster_respawn_timer_ratio` | 刷新倒计时百分比 | **刷新时机预判** |
| `monster_value_score` | 综合价值评估 | 资源重要性 |

#### 4.2 英雄与野怪交互特征 (12维)
| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `hero_dist_to_monster` | 我方英雄到野怪距离 | 争夺位置优势 |
| `dist_advantage_for_monster` | `敌方距离 - 我方距离` | **争夺优势评估** |
| `time_to_kill_monster_by_hero` | `野怪血量 / 英雄攻击力` | 击杀时间预测 |
| `steal_competition_level` | 抢夺竞争激烈程度 | 争夺风险评估 |

#### 4.3 战略与机会成本特征 (12维)
| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `lane_push_advantage` | 兵线推进优势评估 | 打野前提条件 |
| `jungle_safety_level` | 野区安全程度 | 风险控制 |
| `gold_loss_on_lane_while_killing_monster` | 打野期间损失的线上经济 | **机会成本量化** |
| `tower_pressure_window` | 推塔压力时间窗口 | 优先级权衡 |

#### 4.4 高级战略特征 (8维)
| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `enemy_likely_to_contest` | 敌方争夺可能性 | 冲突预测 |
| `optimal_timing_window` | 最优时机窗口 | 时机选择 |
| `psychological_pressure_on_enemy` | 对敌方的心理压力 | 心理战术 |
| `chain_reaction_benefit` | 连锁反应收益 | 长期价值 |

---

### 5. 🏗️ 基础环境特征模块 (148维)
*实现位置: `agent_ppo/feature/feature_process/organ_process.py` - 基础特征*

#### 5.1 基础防御塔特征 (36维)
| 特征类别 | 维度 | 核心功能 |
|----------|------|----------|
| 塔内在特征 | 8维 | 血量、攻击状态、攻击力、攻速 |
| 塔英雄交互 | 10维 | 距离、范围、仇恨、威胁等级 |
| 塔小兵交互 | 8维 | 塔下小兵、偷塔保护、仇恨目标 |
| 塔预测特征 | 10维 | 摧毁时间、DPS承受、危机等级 |

#### 5.2 经济优势特征 (6维)
| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `gold_advantage` | 金币差距 | 装备优势预判 |
| `exp_advantage` | 经验差距 | 等级优势评估 |
| `item_power_diff` | 装备战力差距 | 战斗力对比 |
| `economic_trend` | 经济发展趋势 | 发育速度对比 |

#### 5.3 游戏阶段特征 (4维)
| 特征名 | 计算方式 | 战术价值 |
|--------|----------|----------|
| `game_phase` | 基于游戏时间的阶段划分 | 战术策略调整 |
| `tempo_advantage` | 节奏优势评估 | 主动权把握 |
| `late_game_scaling` | 后期成长潜力 | 长期规划 |

#### 5.4 基础兵线特征 (32维)
| 特征类别 | 维度 | 核心功能 |
|----------|------|----------|
| 兵线宏观状态 | 8维 | 血量、数量、炮车状态 |
| 兵线空间特征 | 8维 | 位置、推进深度、交战点 |
| 兵线目标意图 | 8维 | 攻击目标、仇恨分配、DPS |
| 兵线补刀特征 | 8维 | 可补刀数量、塔补刀预测 |

---

## 🎯 特征使用策略与组合

### 战术组合示例

#### 1. 控线战术组合
```python
if (wave_freeze_potential_score > 0.7 and 
    hero_threat_to_freeze < 0.3 and 
    wave_ranged_minion_advantage >= 0):
    # 执行控线：只补最后一刀，保持兵线位置
    action = "freeze_lane"
```

#### 2. 越塔判断组合
```python
if (hero_dive_survivability_margin > 0 and
    ally_dive_execution_window > 0.6 and
    enemy_hero_dive_threat_score < 0.5):
    # 执行越塔：有生存空间且执行窗口良好
    action = "tower_dive"
```

#### 3. 野怪争夺组合
```python
if (monster_is_alive and
    dist_advantage_for_monster > 2000 and
    lane_push_advantage > 0.6 and
    jungle_safety_level > 0.7):
    # 争夺野怪：位置优势+兵线支持+安全环境
    action = "contest_jungle"
```

#### 4. 兵线运营组合
```python
if (slow_push_identification > 0.6 and
    wave_convergence_timing < 0.3 and
    tower_resource_investment_roi > 0.7):
    # 慢推转强推：等待兵线汇合后推塔
    action = "slow_to_fast_push"
```

---

## 📊 特征权重与归一化

### 归一化策略
1. **距离特征**: 除以地图最大距离(~20000)
2. **时间特征**: 除以最大预估时间(如60秒)
3. **比例特征**: 直接使用[0,1]范围
4. **计数特征**: 除以最大可能数量
5. **优势特征**: 映射到[-1,1]再转换为[0,1]

### 特征重要性分级
- **S级 (权重1.0)**: 生存空间、控线潜力、一刀击杀数
- **A级 (权重0.8)**: 威胁评估、距离优势、资源ROI
- **B级 (权重0.6)**: 心理优势、时序特征、预测特征
- **C级 (权重0.4)**: 辅助特征、占位特征

---

## 🔧 技术实现细节

### 代码架构
```
agent_ppo/feature/feature_process/
├── hero_process.py           # 英雄核心特征 (132维)
├── organ_process.py          # 环境特征主模块 (498维)
│   ├── 基础环境特征         # 148维
│   ├── 野怪特征             # 40维
│   ├── 高级防御塔特征       # 50维
│   └── 专家级兵线特征       # 60维
└── wave_process.py           # 辅助兵线处理 (200维)
```

### 配置文件
- `hero_feature_config.ini`: 英雄特征配置
- `organ_feature_config.ini`: 环境特征配置
- `wave_feature_config.ini`: 兵线特征配置

### 数据流向
```
frame_state → 特征处理器 → 630维特征向量 → 神经网络 → 动作输出
```

---

## 🚀 性能优化建议

### 1. 训练策略
- **阶段训练**: 先基础特征，后高级特征
- **课程学习**: 从简单场景到复杂对抗
- **多任务学习**: 同时优化多个子目标

### 2. 特征选择
- **重要性排序**: 基于梯度重要性筛选核心特征
- **相关性分析**: 移除高度相关的冗余特征
- **在线调整**: 根据训练效果动态调整特征权重

### 3. 计算优化
- **向量化计算**: 批量处理提升效率
- **缓存机制**: 重复计算结果缓存
- **增量更新**: 只更新变化的特征

---

## 📈 预期效果与评估

### AI能力提升预期
| 能力维度 | 基线水平 | 预期提升 | 目标水平 |
|----------|----------|----------|----------|
| **补刀精确度** | 60% | +25% | 85% |
| **越塔决策** | 30% | +50% | 80% |
| **兵线控制** | 20% | +70% | 90% |
| **资源争夺** | 40% | +35% | 75% |
| **整体胜率** | 50% | +30% | 80% |

### 评估指标
1. **对战胜率**: 与不同水平AI的胜负比
2. **经济效率**: 每分钟金币和经验获取
3. **战术执行**: 控线、越塔、野怪争夺成功率
4. **生存能力**: 死亡次数和危险处理能力
5. **决策质量**: 关键时刻的选择正确性

---

## 🛠️ 部署与维护

### 配置参数调优
根据实际游戏数据调整以下关键参数：
- 塔攻击范围: `TOWER_ATTACK_RANGE = 8000`
- 兵线刷新间隔: `WAVE_SPAWN_INTERVAL = 30 * 60`
- 小兵配置ID: `ranged_minion_ids`, `cannon_minion_ids`
- 伤害计算公式: 技能伤害、护甲减免等

### 监控指标
- **特征计算时间**: 每帧特征提取耗时
- **内存使用**: 特征缓存内存占用
- **特征分布**: 各特征值的统计分布
- **异常检测**: 特征值异常波动监控

---

## 📚 版本历史

### v1.0 - 基础特征体系
- 实现英雄、塔、兵线基础特征
- 总计280维特征

### v2.0 - 高级战术特征
- 新增野怪控制特征
- 实现高级防御塔特征
- 总计430维特征

### v3.0 - 专家级运营特征 (当前版本)
- 实现专家级兵线特征
- 完整心理博弈系统
- **总计630维特征**

---

## 🎯 未来发展方向

### 短期优化 (1-2月)
1. **特征效果验证**: A/B测试验证特征价值
2. **性能优化**: 提升特征计算效率
3. **参数调优**: 根据训练效果调整权重

### 中期扩展 (3-6月)
1. **多英雄适配**: 扩展到不同英雄类型
2. **地图泛化**: 适配不同1v1地图
3. **对手建模**: 增加对手行为预测特征

### 长期愿景 (6月+)
1. **5v5扩展**: 扩展到团队对战
2. **实时对抗**: 与人类选手实时对战
3. **自适应学习**: 在线学习和策略调整

---

## 📧 联系与支持

本特征工程文档涵盖了完整的630维特征体系，为训练出职业级1v1王者荣耀AI提供了坚实的技术基础。

**特征总结**:
- 🎯 **630维综合特征** - 覆盖游戏的每个关键维度
- 🧠 **专家级战术** - 从操作到运营的全面提升  
- ⚡ **实时决策** - 毫秒级的精确判断
- 🏆 **职业水准** - 对标人类顶级选手

通过这套完整的特征体系，AI将具备：职业选手级的补刀精度、大师级的兵线运营、专家级的风险评估、以及战术大师级的决策能力！
