#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Enhanced by Deep RL Analysis

全面的孙尚香1v1特征处理器
基于深度强化学习分析，实现完整的状态空间表示

特征设计原则：
1. 全面性：覆盖所有影响决策的关键信息
2. 相对性：重视相对特征而非绝对值
3. 稠密性：提供足够细粒度的状态描述
4. 归一化：所有特征归一化到合理范围
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict


class ComprehensiveFeatureProcessor:
    """全面的特征处理器 - 基于深度RL分析的完整状态空间"""
    
    def __init__(self, camp: str):
        self.main_camp = camp
        self.enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 历史状态追踪 (用于计算变化量)
        self.state_history = deque(maxlen=10)
        self.last_frame_data = None
        
        # 特征维度定义
        self.feature_dimensions = {
            'self_hero_state': 25,      # 我方英雄状态
            'enemy_hero_state': 15,     # 敌方英雄状态  
            'environment_state': 20,    # 环境状态
            'relational_features': 30,  # 相对/组合特征
            'temporal_features': 15,    # 时序特征
            'strategic_features': 20    # 战略特征
        }
        
        # 孙尚香技能配置
        self.sunshangxiang_skills = {
            'passive': 11100,    # 被动：活力迸发
            'skill1': 11110,     # 一技能：翻滚突袭
            'skill2': 11120,     # 二技能：红莲爆弹
            'skill3': 11130,     # 三技能：究极弩炮
            'summoner': 80115    # 召唤师技能：闪现
        }
        
    def extract_comprehensive_features(self, observation: Dict, frame_no: int) -> np.ndarray:
        """提取全面的特征向量"""
        frame_state = observation["frame_state"]
        
        # 获取英雄状态
        my_hero = self._find_hero_by_camp(frame_state["hero_states"], self.main_camp)
        enemy_hero = self._find_hero_by_camp(frame_state["hero_states"], self.enemy_camp)
        
        if not my_hero or not enemy_hero:
            return np.zeros(sum(self.feature_dimensions.values()))
        
        all_features = []
        
        # 1. 我方英雄状态 (25维)
        self_features = self._extract_self_hero_features(my_hero, frame_state)
        all_features.extend(self_features)
        
        # 2. 敌方英雄状态 (15维)
        enemy_features = self._extract_enemy_hero_features(enemy_hero, frame_state)
        all_features.extend(enemy_features)
        
        # 3. 环境状态 (20维)
        env_features = self._extract_environment_features(frame_state, my_hero)
        all_features.extend(env_features)
        
        # 4. 相对/组合特征 (30维)
        relational_features = self._extract_relational_features(my_hero, enemy_hero, frame_state)
        all_features.extend(relational_features)
        
        # 5. 时序特征 (15维)
        temporal_features = self._extract_temporal_features(my_hero, enemy_hero, frame_no)
        all_features.extend(temporal_features)
        
        # 6. 战略特征 (20维)
        strategic_features = self._extract_strategic_features(my_hero, enemy_hero, frame_state)
        all_features.extend(strategic_features)
        
        # 更新历史状态
        self._update_state_history(my_hero, enemy_hero, frame_state, frame_no)
        
        return np.array(all_features, dtype=np.float32)
    
    def _extract_self_hero_features(self, my_hero: Dict, frame_state: Dict) -> List[float]:
        """提取我方英雄状态特征 (25维)"""
        features = []
        actor_state = my_hero.get("actor_state", {})
        values = actor_state.get("values", {})
        
        # 核心属性 (8维)
        hp_ratio = float(actor_state.get("hp", 0)) / max(float(actor_state.get("max_hp", 1)), 1)
        features.append(hp_ratio)
        
        mp_ratio = float(values.get("ep", 0)) / max(float(values.get("max_ep", 1)), 1)
        features.append(mp_ratio)
        
        level = float(my_hero.get("level", 1)) / 15.0  # 归一化到[0,1]
        features.append(level)
        
        exp_ratio = float(my_hero.get("exp", 0)) / max(self._get_exp_required_for_level(my_hero.get("level", 1)), 1)
        features.append(exp_ratio)
        
        gold = float(my_hero.get("money", 0)) / 10000.0  # 归一化
        features.append(gold)
        
        total_gold = float(my_hero.get("moneyCnt", 0)) / 15000.0  # 归一化
        features.append(total_gold)
        
        kill_count = float(my_hero.get("killCnt", 0)) / 10.0  # 归一化
        features.append(kill_count)
        
        death_count = float(my_hero.get("deadCnt", 0)) / 10.0  # 归一化
        features.append(death_count)
        
        # 战斗属性 (6维)
        phy_atk = float(values.get("phy_atk", 0)) / 500.0  # 归一化
        features.append(phy_atk)
        
        phy_def = float(values.get("phy_def", 0)) / 300.0  # 归一化
        features.append(phy_def)
        
        atk_spd = float(values.get("atk_spd", 0)) / 200.0  # 归一化
        features.append(atk_spd)
        
        mov_spd = float(values.get("mov_spd", 0)) / 500.0  # 归一化
        features.append(mov_spd)
        
        attack_range = float(actor_state.get("attack_range", 0)) / 1000.0  # 归一化
        features.append(attack_range)
        
        crit_rate = float(values.get("crit_rate", 0)) / 100.0  # 归一化
        features.append(crit_rate)
        
        # 技能状态 (8维)
        skill_features = self._extract_skill_features(my_hero)
        features.extend(skill_features)
        
        # 位置信息 (2维)
        location = actor_state.get("location", {"x": 0, "z": 0})
        x_norm = float(location.get("x", 0)) / 15000.0  # 地图归一化
        z_norm = float(location.get("z", 0)) / 15000.0  # 地图归一化
        features.extend([x_norm, z_norm])
        
        # Buff状态 (1维) - 简化处理
        buff_count = len(my_hero.get("buff_state", {}).get("buff_skills", []))
        features.append(min(buff_count / 5.0, 1.0))
        
        return features
    
    def _extract_enemy_hero_features(self, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """提取敌方英雄状态特征 (15维)"""
        features = []
        actor_state = enemy_hero.get("actor_state", {})
        values = actor_state.get("values", {})
        
        # 可见的核心属性 (6维)
        hp_ratio = float(actor_state.get("hp", 0)) / max(float(actor_state.get("max_hp", 1)), 1)
        features.append(hp_ratio)
        
        level = float(enemy_hero.get("level", 1)) / 15.0
        features.append(level)
        
        kill_count = float(enemy_hero.get("killCnt", 0)) / 10.0
        features.append(kill_count)
        
        death_count = float(enemy_hero.get("deadCnt", 0)) / 10.0
        features.append(death_count)
        
        # 位置信息 (2维)
        location = actor_state.get("location", {"x": 0, "z": 0})
        x_norm = float(location.get("x", 0)) / 15000.0
        z_norm = float(location.get("z", 0)) / 15000.0
        features.extend([x_norm, z_norm])
        
        # 可见性 (1维)
        is_visible = 1.0 if self._is_enemy_visible(enemy_hero, frame_state) else 0.0
        features.append(is_visible)
        
        # 推断的技能状态 (6维) - 基于等级和时间推断
        inferred_skills = self._infer_enemy_skill_status(enemy_hero)
        features.extend(inferred_skills)
        
        return features
    
    def _extract_environment_features(self, frame_state: Dict, my_hero: Dict) -> List[float]:
        """提取环境状态特征 (20维)"""
        features = []
        npcs = frame_state.get("npc_states", [])
        my_pos = self._get_position(my_hero)
        
        # 兵线信息 (8维)
        minion_features = self._extract_minion_features(npcs, my_pos)
        features.extend(minion_features)
        
        # 防御塔信息 (4维)
        tower_features = self._extract_tower_features(npcs)
        features.extend(tower_features)
        
        # 地图元素 (4维)
        map_features = self._extract_map_elements(frame_state, my_pos)
        features.extend(map_features)
        
        # 环境风险评估 (4维)
        risk_features = self._extract_environment_risks(npcs, my_pos)
        features.extend(risk_features)
        
        return features
    
    def _extract_relational_features(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """提取相对/组合特征 (30维) - 这是最重要的特征类别"""
        features = []
        
        # 经济相关 (6维)
        my_money = float(my_hero.get("money", 0))
        enemy_money = float(enemy_hero.get("money", 0))
        my_total = float(my_hero.get("moneyCnt", 0))
        
        money_delta = (my_money - enemy_money) / 5000.0  # 归一化
        features.append(money_delta)
        
        money_ratio = my_money / max(enemy_money, 1.0)
        features.append(min(money_ratio, 3.0) / 3.0)  # 归一化到[0,1]
        
        total_money_advantage = (my_total - float(enemy_hero.get("moneyCnt", 0))) / 10000.0
        features.append(total_money_advantage)
        
        # 经济效率
        game_time = max(frame_state.get("frameNo", 0) / 1800.0, 1.0)  # 分钟
        my_gold_per_min = my_total / game_time / 2000.0  # 归一化
        features.append(min(my_gold_per_min, 1.0))
        
        # 经济增长趋势 (需要历史数据)
        money_growth_rate = self._calculate_money_growth_rate(my_hero)
        features.append(money_growth_rate)
        
        # 装备价值差
        equipment_advantage = self._calculate_equipment_advantage(my_hero, enemy_hero)
        features.append(equipment_advantage)
        
        # 等级经验相关 (4维)
        level_diff = (float(my_hero.get("level", 1)) - float(enemy_hero.get("level", 1))) / 15.0
        features.append(level_diff)
        
        exp_advantage = self._calculate_exp_advantage(my_hero, enemy_hero)
        features.append(exp_advantage)
        
        # 等级压制程度
        level_suppress = max(0.0, level_diff) * 2.0  # 放大等级优势
        features.append(min(level_suppress, 1.0))
        
        # 升级接近度
        my_exp_to_levelup = self._get_exp_to_next_level(my_hero) / 1000.0
        features.append(min(my_exp_to_levelup, 1.0))
        
        # 血量对比 (4维)
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        
        hp_advantage = my_hp_ratio - enemy_hp_ratio
        features.append(hp_advantage)
        
        hp_ratio_product = my_hp_ratio * enemy_hp_ratio  # 双方血量乘积，反映总体血量
        features.append(hp_ratio_product)
        
        # 血量威胁度 (我方低血量时的危险程度)
        hp_threat = max(0.0, 0.5 - my_hp_ratio) * 2.0
        features.append(hp_threat)
        
        # 击杀机会 (敌方低血量时的机会)
        kill_opportunity = max(0.0, 0.3 - enemy_hp_ratio) * 3.0
        features.append(min(kill_opportunity, 1.0))
        
        # 距离相关 (6维)
        distance = self._calculate_distance(my_hero, enemy_hero)
        actor_state = my_hero.get("actor_state", {})
        if isinstance(actor_state, dict):
            attack_range = float(actor_state.get("attack_range", 600))
        else:
            attack_range = 600.0
        
        normalized_distance = distance / 2000.0  # 归一化
        features.append(min(normalized_distance, 1.0))
        
        # 攻击范围内
        in_attack_range = 1.0 if distance <= attack_range else 0.0
        features.append(in_attack_range)
        
        # 安全距离评估
        safe_distance = 1.0 if distance > 800 else 0.0
        features.append(safe_distance)
        
        # 追击距离 (在追击范围内)
        chase_distance = 1.0 if distance <= attack_range * 1.5 else 0.0
        features.append(chase_distance)
        
        # 相对位置优势
        position_advantage = self._calculate_position_advantage(my_hero, enemy_hero, frame_state)
        features.append(position_advantage)
        
        # 机动性对比
        mobility_advantage = self._calculate_mobility_advantage(my_hero, enemy_hero)
        features.append(mobility_advantage)
        
        # 战斗力评估 (4维)
        my_combat_power = self._estimate_combat_power(my_hero)
        enemy_combat_power = self._estimate_combat_power(enemy_hero)
        
        combat_power_ratio = my_combat_power / max(enemy_combat_power, 0.1)
        features.append(min(combat_power_ratio / 2.0, 1.0))  # 归一化
        
        # 爆发潜力对比
        my_burst_potential = self._calculate_burst_potential(my_hero)
        enemy_burst_potential = self._calculate_burst_potential(enemy_hero)
        burst_advantage = (my_burst_potential - enemy_burst_potential) / 2.0
        features.append(burst_advantage)
        
        # 持续作战能力
        sustain_advantage = self._calculate_sustain_advantage(my_hero, enemy_hero)
        features.append(sustain_advantage)
        
        # 技能优势窗口
        skill_window_advantage = self._calculate_skill_window_advantage(my_hero, enemy_hero)
        features.append(skill_window_advantage)
        
        return features
    
    def _extract_temporal_features(self, my_hero: Dict, enemy_hero: Dict, frame_no: int) -> List[float]:
        """提取时序特征 (15维)"""
        features = []
        
        if len(self.state_history) < 2:
            return [0.0] * 15
        
        current_state = {
            'my_hp': self._get_hp_ratio(my_hero),
            'enemy_hp': self._get_hp_ratio(enemy_hero),
            'my_money': float(my_hero.get("money", 0)),
            'enemy_money': float(enemy_hero.get("money", 0)),
            'distance': self._calculate_distance(my_hero, enemy_hero),
            'my_pos': self._get_position(my_hero),
            'enemy_pos': self._get_position(enemy_hero)
        }
        
        prev_state = self.state_history[-1]
        
        # 血量变化趋势 (4维)
        my_hp_change = current_state['my_hp'] - prev_state['my_hp']
        enemy_hp_change = current_state['enemy_hp'] - prev_state['enemy_hp']
        features.extend([my_hp_change, enemy_hp_change])
        
        # 血量变化优势
        hp_change_advantage = my_hp_change - enemy_hp_change
        features.append(hp_change_advantage)
        
        # 血量变化速率 (正值表示回复，负值表示受伤)
        hp_change_rate = abs(my_hp_change) + abs(enemy_hp_change)
        features.append(hp_change_rate)
        
        # 经济变化趋势 (4维)
        my_money_change = (current_state['my_money'] - prev_state['my_money']) / 100.0
        enemy_money_change = (current_state['enemy_money'] - prev_state['enemy_money']) / 100.0
        features.extend([my_money_change, enemy_money_change])
        
        money_change_advantage = my_money_change - enemy_money_change
        features.append(money_change_advantage)
        
        # 发育效率
        farming_efficiency = my_money_change / max(abs(my_hp_change) + 0.01, 0.01)  # 收益/风险比
        features.append(min(farming_efficiency, 5.0) / 5.0)
        
        # 位置变化 (4维)
        my_pos_change = math.sqrt(
            (current_state['my_pos'][0] - prev_state['my_pos'][0])**2 +
            (current_state['my_pos'][1] - prev_state['my_pos'][1])**2
        ) / 500.0  # 归一化
        features.append(min(my_pos_change, 1.0))
        
        enemy_pos_change = math.sqrt(
            (current_state['enemy_pos'][0] - prev_state['enemy_pos'][0])**2 +
            (current_state['enemy_pos'][1] - prev_state['enemy_pos'][1])**2
        ) / 500.0
        features.append(min(enemy_pos_change, 1.0))
        
        # 距离变化
        distance_change = (current_state['distance'] - prev_state['distance']) / 500.0
        features.append(distance_change)
        
        # 相对移动速度
        relative_mobility = my_pos_change - enemy_pos_change
        features.append(relative_mobility)
        
        # 游戏阶段 (3维)
        game_progress = min(frame_no / 18000.0, 1.0)  # 10分钟游戏
        features.append(game_progress)
        
        # 早中后期标识
        early_game = 1.0 if game_progress < 0.3 else 0.0
        late_game = 1.0 if game_progress > 0.7 else 0.0
        features.extend([early_game, late_game])
        
        return features
    
    def _extract_strategic_features(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """提取战略特征 (20维)"""
        features = []
        
        # 推塔机会 (4维)
        tower_opportunity = self._assess_tower_opportunity(my_hero, enemy_hero, frame_state)
        features.extend(tower_opportunity)
        
        # 发育机会 (4维)
        farming_opportunity = self._assess_farming_opportunity(my_hero, frame_state)
        features.extend(farming_opportunity)
        
        # 击杀机会 (4维)
        kill_opportunity = self._assess_kill_opportunity(my_hero, enemy_hero)
        features.extend(kill_opportunity)
        
        # 风险评估 (4维)
        risk_assessment = self._assess_strategic_risks(my_hero, enemy_hero, frame_state)
        features.extend(risk_assessment)
        
        # 控制权 (4维)
        map_control = self._assess_map_control(my_hero, enemy_hero, frame_state)
        features.extend(map_control)
        
        return features
    
    # ============ 辅助方法 ============
    
    def _find_hero_by_camp(self, hero_states: List, camp: str) -> Optional[Dict]:
        """根据阵营查找英雄"""
        if not isinstance(hero_states, list):
            return None
        
        for hero in hero_states:
            if not isinstance(hero, dict):
                continue
            
            actor_state = hero.get("actor_state", {})
            if isinstance(actor_state, dict) and actor_state.get("camp") == camp:
                return hero
        return None
    
    def _get_position(self, hero: Dict) -> Tuple[float, float]:
        """获取英雄位置"""
        if not isinstance(hero, dict):
            return (0.0, 0.0)
        
        actor_state = hero.get("actor_state", {})
        if not isinstance(actor_state, dict):
            return (0.0, 0.0)
        
        location = actor_state.get("location", {})
        if not isinstance(location, dict):
            return (0.0, 0.0)
        
        x = location.get("x", 0)
        z = location.get("z", 0)
        return (float(x) if x is not None else 0.0, float(z) if z is not None else 0.0)
    
    def _get_hp_ratio(self, hero: Dict) -> float:
        """获取血量比例"""
        if not isinstance(hero, dict):
            return 0.0
        
        actor_state = hero.get("actor_state", {})
        if not isinstance(actor_state, dict):
            return 0.0
        
        try:
            hp = float(actor_state.get("hp", 0))
            max_hp = float(actor_state.get("max_hp", 1))
            return hp / max(max_hp, 1.0)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_distance(self, hero1: Dict, hero2: Dict) -> float:
        """计算两个英雄之间的距离"""
        pos1 = self._get_position(hero1)
        pos2 = self._get_position(hero2)
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _safe_get_nested_value(self, obj: Dict, keys: List[str], default=0.0) -> float:
        """安全地获取嵌套字典的值"""
        if not isinstance(obj, dict):
            return float(default)
        
        current = obj
        for key in keys:
            if not isinstance(current, dict):
                return float(default)
            current = current.get(key, {})
        
        try:
            return float(current) if current is not None else float(default)
        except (ValueError, TypeError):
            return float(default)
    
    def _extract_skill_features(self, hero: Dict) -> List[float]:
        """提取技能状态特征 (8维)"""
        features = []
        skill_state = hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        
        # 前4个技能槽 (每个2维: 可用性, CD比例)
        for i in range(4):
            if i < len(slots):
                slot = slots[i]
                usable = 1.0 if slot.get("usable", False) else 0.0
                cd_ratio = float(slot.get("cooldown", 0)) / max(float(slot.get("cooldown_max", 1)), 1)
                features.extend([usable, cd_ratio])
            else:
                features.extend([0.0, 0.0])
        
        return features
    
    def _extract_minion_features(self, npcs: List, my_pos: Tuple[float, float]) -> List[float]:
        """提取小兵特征 (8维)"""
        features = []
        
        my_minions = 0
        enemy_minions = 0
        nearby_low_hp_minions = 0
        minion_advantage_position = 0.0
        
        for npc in npcs:
            if 'SOLDIER' in npc.get('sub_type', ''):
                npc_pos = npc.get('location', {})
                if npc_pos:
                    npc_position = (float(npc_pos.get('x', 0)), float(npc_pos.get('z', 0)))
                    dist = math.sqrt((my_pos[0] - npc_position[0])**2 + (my_pos[1] - npc_position[1])**2)
                    
                    if npc.get('camp') == self.main_camp:
                        my_minions += 1
                        if dist <= 2000:  # 附近我方小兵
                            minion_advantage_position += 0.1
                    else:
                        enemy_minions += 1
                        if dist <= 1200:  # 可攻击范围内敌方小兵
                            hp_ratio = float(npc.get('hp', 0)) / max(float(npc.get('max_hp', 1)), 1)
                            if hp_ratio < 0.3:  # 低血量，可补刀
                                nearby_low_hp_minions += 1
        
        features.extend([
            min(my_minions / 5.0, 1.0),
            min(enemy_minions / 5.0, 1.0),
            (my_minions - enemy_minions) / 10.0,  # 小兵数量优势
            min(nearby_low_hp_minions / 3.0, 1.0),
            min(minion_advantage_position, 1.0),
            0.0, 0.0, 0.0  # 预留
        ])
        
        return features
    
    def _extract_tower_features(self, npcs: List) -> List[float]:
        """提取防御塔特征 (4维)"""
        features = []
        
        my_tower_hp = 0.0
        enemy_tower_hp = 0.0
        
        for npc in npcs:
            if 'TOWER' in npc.get('sub_type', ''):
                hp_ratio = float(npc.get('hp', 0)) / max(float(npc.get('max_hp', 1)), 1)
                if npc.get('camp') == self.main_camp:
                    my_tower_hp = hp_ratio
                else:
                    enemy_tower_hp = hp_ratio
        
        tower_hp_advantage = my_tower_hp - enemy_tower_hp
        
        features.extend([
            my_tower_hp,
            enemy_tower_hp,
            tower_hp_advantage,
            0.0  # 预留
        ])
        
        return features
    
    def _update_state_history(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict, frame_no: int):
        """更新状态历史"""
        current_state = {
            'frame_no': frame_no,
            'my_hp': self._get_hp_ratio(my_hero),
            'enemy_hp': self._get_hp_ratio(enemy_hero),
            'my_money': float(my_hero.get("money", 0)),
            'enemy_money': float(enemy_hero.get("money", 0)),
            'distance': self._calculate_distance(my_hero, enemy_hero),
            'my_pos': self._get_position(my_hero),
            'enemy_pos': self._get_position(enemy_hero)
        }
        
        self.state_history.append(current_state)
    
    # ============ 复杂计算方法 ============
    
    def _get_exp_required_for_level(self, level: int) -> int:
        """获取升级所需经验 - 简化计算"""
        base_exp = 100
        return base_exp * (level ** 1.5)
    
    def _get_exp_to_next_level(self, hero: Dict) -> float:
        """获取距离下一级的经验"""
        current_level = hero.get("level", 1)
        current_exp = float(hero.get("exp", 0))
        required_exp = self._get_exp_required_for_level(current_level + 1)
        return max(0.0, required_exp - current_exp)
    
    def _calculate_money_growth_rate(self, hero: Dict) -> float:
        """计算经济增长率"""
        if len(self.state_history) < 2:
            return 0.0
        
        current_money = float(hero.get("money", 0))
        prev_money = self.state_history[-1]['my_money']
        growth_rate = (current_money - prev_money) / max(prev_money, 100.0)
        
        return min(growth_rate, 1.0)
    
    def _calculate_equipment_advantage(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算装备优势 - 基于总经济估算"""
        my_total = float(my_hero.get("moneyCnt", 0))
        enemy_total = float(enemy_hero.get("moneyCnt", 0))
        
        # 简化：假设70%经济转化为装备价值
        my_equipment_value = my_total * 0.7
        enemy_equipment_value = enemy_total * 0.7
        
        advantage = (my_equipment_value - enemy_equipment_value) / 10000.0
        return max(-1.0, min(1.0, advantage))
    
    def _calculate_exp_advantage(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算经验优势"""
        my_exp = float(my_hero.get("exp", 0))
        enemy_exp = float(enemy_hero.get("exp", 0))
        
        exp_diff = (my_exp - enemy_exp) / 1000.0
        return max(-1.0, min(1.0, exp_diff))
    
    def _calculate_position_advantage(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> float:
        """计算位置优势"""
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 简化：基于与地图中心的距离
        map_center = (7500.0, 7500.0)  # 假设地图大小15000x15000
        
        my_center_dist = math.sqrt((my_pos[0] - map_center[0])**2 + (my_pos[1] - map_center[1])**2)
        enemy_center_dist = math.sqrt((enemy_pos[0] - map_center[0])**2 + (enemy_pos[1] - map_center[1])**2)
        
        # 更接近中心位置更有利
        position_advantage = (enemy_center_dist - my_center_dist) / 5000.0
        return max(-1.0, min(1.0, position_advantage))
    
    def _calculate_mobility_advantage(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算机动性优势"""
        my_speed = self._safe_get_nested_value(my_hero, ["actor_state", "values", "mov_spd"], 0)
        enemy_speed = self._safe_get_nested_value(enemy_hero, ["actor_state", "values", "mov_spd"], 0)
        
        mobility_advantage = (my_speed - enemy_speed) / 500.0
        return max(-1.0, min(1.0, mobility_advantage))
    
    def _estimate_combat_power(self, hero: Dict) -> float:
        """估算战斗力"""
        values = hero.get("actor_state", {}).get("values", {})
        level = float(hero.get("level", 1))
        
        # 简化的战斗力计算
        phy_atk = float(values.get("phy_atk", 0))
        atk_spd = float(values.get("atk_spd", 0))
        phy_def = float(values.get("phy_def", 0))
        hp_ratio = self._get_hp_ratio(hero)
        
        combat_power = (phy_atk * (1 + atk_spd / 100.0) + phy_def) * level * hp_ratio
        return combat_power / 1000.0  # 归一化
    
    def _calculate_burst_potential(self, hero: Dict) -> float:
        """计算爆发潜力"""
        # 基于技能可用性和攻击力
        skill_state = hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        
        available_skills = sum(1 for slot in slots if slot.get("usable", False))
        phy_atk = self._safe_get_nested_value(hero, ["actor_state", "values", "phy_atk"], 0)
        
        burst_potential = (available_skills * phy_atk) / 2000.0
        return min(burst_potential, 1.0)
    
    def _calculate_sustain_advantage(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算持续作战能力优势"""
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        
        my_ep = self._safe_get_nested_value(my_hero, ["actor_state", "values", "ep"], 0)
        my_max_ep = self._safe_get_nested_value(my_hero, ["actor_state", "values", "max_ep"], 1)
        my_mp_ratio = my_ep / max(my_max_ep, 1)
        
        enemy_ep = self._safe_get_nested_value(enemy_hero, ["actor_state", "values", "ep"], 0)
        enemy_max_ep = self._safe_get_nested_value(enemy_hero, ["actor_state", "values", "max_ep"], 1)
        enemy_mp_ratio = enemy_ep / max(enemy_max_ep, 1)
        
        # 综合血量和蓝量的持续作战能力
        my_sustain = (my_hp_ratio + my_mp_ratio) / 2.0
        enemy_sustain = (enemy_hp_ratio + enemy_mp_ratio) / 2.0
        
        return my_sustain - enemy_sustain
    
    def _calculate_skill_window_advantage(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算技能窗口优势"""
        # 简化：基于技能可用性
        my_skills = self._extract_skill_features(my_hero)
        enemy_skills = self._extract_skill_features(enemy_hero) if enemy_hero else [0.0] * 8
        
        my_available = sum(my_skills[i] for i in range(0, 8, 2))  # 可用技能数
        enemy_available = sum(enemy_skills[i] for i in range(0, 8, 2))
        
        return (my_available - enemy_available) / 4.0
    
    # ============ 简化的辅助方法 ============
    
    def _is_enemy_visible(self, enemy_hero: Dict, frame_state: Dict) -> bool:
        """判断敌方英雄是否可见 - 简化实现"""
        return True  # 简化处理，假设总是可见
    
    def _infer_enemy_skill_status(self, enemy_hero: Dict) -> List[float]:
        """推断敌方技能状态 (6维) - 简化实现"""
        return [0.5] * 6  # 简化处理，假设技能状态未知
    
    def _extract_map_elements(self, frame_state: Dict, my_pos: Tuple[float, float]) -> List[float]:
        """提取地图元素特征 (4维) - 简化实现"""
        return [0.0] * 4
    
    def _extract_environment_risks(self, npcs: List, my_pos: Tuple[float, float]) -> List[float]:
        """提取环境风险 (4维) - 简化实现"""
        return [0.0] * 4
    
    def _assess_tower_opportunity(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """评估推塔机会 (4维) - 简化实现"""
        return [0.0] * 4
    
    def _assess_farming_opportunity(self, my_hero: Dict, frame_state: Dict) -> List[float]:
        """评估发育机会 (4维) - 简化实现"""
        return [0.0] * 4
    
    def _assess_kill_opportunity(self, my_hero: Dict, enemy_hero: Dict) -> List[float]:
        """评估击杀机会 (4维) - 简化实现"""
        return [0.0] * 4
    
    def _assess_strategic_risks(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """评估战略风险 (4维) - 简化实现"""
        return [0.0] * 4
    
    def _assess_map_control(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """评估地图控制权 (4维) - 简化实现"""
        return [0.0] * 4
