#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors (Enhanced for SunShangxiang)

孙尚香专用增强特征处理器
基于三篇论文的核心洞察设计：
1. 分层动作空间的特征表示
2. 技能连招时序建模
3. 射手英雄专用战术特征
"""

import numpy as np
import math
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional


class SunShangxiangEnhancedProcessor:
    """孙尚香专用增强特征处理器"""
    
    def __init__(self, camp: str):
        self.main_camp = camp
        self.enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 孙尚香技能ID定义
        self.SKILL_IDS = {
            'passive': 11100,    # 被动：活力迸发
            'skill1': 11110,     # 一技能：翻滚突袭
            'skill2': 11120,     # 二技能：红莲爆弹
            'skill3': 11130,     # 三技能：究极弩炮
            'summoner': 80115    # 召唤师技能：闪现
        }
        
        # 连招检测器
        self.combo_detector = SunShangxiangComboDetector()
        
        # 战术分析器
        self.tactical_analyzer = MarksmanTacticalAnalyzer()
        
        # 时序状态追踪
        self.state_tracker = StateTracker()
        
    def extract_enhanced_features(self, observation: Dict, frame_no: int) -> List[float]:
        """提取孙尚香专用增强特征"""
        frame_state = observation["frame_state"]
        
        # 获取英雄状态
        my_hero = self._find_hero_by_camp(frame_state["hero_states"], self.main_camp)
        enemy_hero = self._find_hero_by_camp(frame_state["hero_states"], self.enemy_camp)
        
        if not my_hero or not enemy_hero:
            return [0.0] * self._get_feature_dimension()
        
        features = []
        
        # 1. 孙尚香技能连招特征 (30维)
        combo_features = self.combo_detector.extract_combo_features(
            my_hero, frame_no, observation.get("legal_action", [])
        )
        features.extend(combo_features)
        
        # 2. 射手战术特征 (25维)
        tactical_features = self.tactical_analyzer.extract_tactical_features(
            my_hero, enemy_hero, frame_state
        )
        features.extend(tactical_features)
        
        # 3. 博弈状态特征 (20维)
        game_state_features = self._extract_game_state_features(
            my_hero, enemy_hero, frame_state
        )
        features.extend(game_state_features)
        
        # 4. 时序动态特征 (15维)
        temporal_features = self.state_tracker.extract_temporal_features(
            my_hero, enemy_hero, frame_no
        )
        features.extend(temporal_features)
        
        # 5. 环境交互特征 (10维)
        interaction_features = self._extract_interaction_features(
            my_hero, frame_state
        )
        features.extend(interaction_features)
        
        return features
    
    def _get_feature_dimension(self) -> int:
        """获取特征总维度"""
        return 30 + 25 + 20 + 15 + 10  # 100维增强特征
    
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
    
    def _extract_game_state_features(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """提取博弈状态特征"""
        features = []
        
        # 血量优势
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        hp_advantage = my_hp_ratio - enemy_hp_ratio
        features.append(hp_advantage)
        
        # 经济优势系统 (增强版)
        economic_features = self._extract_economic_features(my_hero, enemy_hero, frame_state)
        features.extend(economic_features[:8])  # 8维经济特征
        
        # 等级优势
        my_level = float(my_hero.get("level", 1))
        enemy_level = float(enemy_hero.get("level", 1))
        level_advantage = (my_level - enemy_level) / 15.0
        features.append(level_advantage)
        
        # 距离关系
        distance = self._calculate_distance(my_hero, enemy_hero)
        normalized_distance = min(distance, 15000.0) / 15000.0
        features.append(normalized_distance)
        
        # 攻击范围优势
        my_range = self._safe_get_nested(my_hero, ["actor_state", "attack_range"], 0)
        enemy_range = self._safe_get_nested(enemy_hero, ["actor_state", "attack_range"], 0)
        range_advantage = (my_range - enemy_range) / 1000.0
        features.append(range_advantage)
        
        # 经济决策状态特征 (新增)
        economic_decision_features = self._extract_economic_decision_features(
            my_hero, enemy_hero, frame_state, distance
        )
        features.extend(economic_decision_features[:8])  # 8维经济决策特征
        
        # 填充到20维
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _extract_economic_features(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """提取经济系统特征"""
        features = []
        
        # 基础经济数据
        my_money = float(my_hero.get("money", 0))
        enemy_money = float(enemy_hero.get("money", 0))
        my_total_money = float(my_hero.get("moneyCnt", 0))  # 总经济
        
        # 零和经济优势 (delta)
        money_delta = my_money - enemy_money
        normalized_delta = money_delta / 10000.0  # 归一化
        features.append(normalized_delta)
        
        # 经济增长率 (需要历史数据支持)
        money_growth_rate = 0.0  # 简化处理
        features.append(money_growth_rate)
        
        # 经济效率 (每分钟经济)
        game_time_minutes = max(frame_state.get("frameNo", 0) / 1800.0, 1.0)  # 30fps * 60s
        economic_efficiency = my_total_money / game_time_minutes / 1000.0  # 归一化
        features.append(min(economic_efficiency, 1.0))
        
        # 装备价值优势
        equipment_value_advantage = self._calculate_equipment_value_advantage(my_hero, enemy_hero)
        features.append(equipment_value_advantage)
        
        # 补刀效率相关
        last_hit_potential = self._calculate_last_hit_potential(my_hero, frame_state)
        features.append(last_hit_potential)
        
        # 经济压制能力
        economic_pressure = 1.0 if money_delta > 2000 else 0.0
        features.append(economic_pressure)
        
        # 经济劣势程度
        economic_disadvantage = max(0.0, -money_delta / 5000.0)
        features.append(min(economic_disadvantage, 1.0))
        
        # 经济追赶潜力
        economic_catchup_potential = self._calculate_catchup_potential(my_hero, enemy_hero, frame_state)
        features.append(economic_catchup_potential)
        
        return features
    
    def _extract_economic_decision_features(self, my_hero: Dict, enemy_hero: Dict, 
                                          frame_state: Dict, distance: float) -> List[float]:
        """提取经济决策状态特征"""
        features = []
        
        # 经济优势
        my_money = float(my_hero.get("money", 0))
        enemy_money = float(enemy_hero.get("money", 0))
        money_delta = my_money - enemy_money
        
        # 攻击距离
        my_attack_range = self._safe_get_nested(my_hero, ["actor_state", "attack_range"], 600)
        
        # 决策状态1: 是否应该优先攻击英雄 (delta > 0 且在攻击范围内)
        should_attack_hero = 1.0 if (money_delta > 0 and distance <= my_attack_range) else 0.0
        features.append(should_attack_hero)
        
        # 决策状态2: 是否应该转向攻击小兵 (delta > 0 但不在攻击范围内)
        should_farm_minions = 1.0 if (money_delta > 0 and distance > my_attack_range) else 0.0
        features.append(should_farm_minions)
        
        # 决策状态3: 是否处于经济劣势需要发育
        need_farming = 1.0 if money_delta < -1000 else 0.0
        features.append(need_farming)
        
        # 决策状态4: 小兵补刀机会
        minion_last_hit_opportunity = self._evaluate_minion_last_hit_opportunity(my_hero, frame_state)
        features.append(minion_last_hit_opportunity)
        
        # 决策状态5: 安全发育状态 (血量足够且有小兵)
        safe_farming_state = self._evaluate_safe_farming_state(my_hero, enemy_hero, frame_state)
        features.append(safe_farming_state)
        
        # 决策状态6: 经济压制机会 (经济领先且可以压制对手发育)
        economic_suppress_opportunity = 1.0 if (money_delta > 2000 and distance <= my_attack_range * 1.2) else 0.0
        features.append(economic_suppress_opportunity)
        
        # 决策状态7: 装备购买时机
        equipment_purchase_timing = self._evaluate_equipment_purchase_timing(my_hero)
        features.append(equipment_purchase_timing)
        
        # 决策状态8: 经济风险评估 (是否应该保守发育)
        economic_risk_level = self._evaluate_economic_risk(my_hero, enemy_hero, frame_state)
        features.append(economic_risk_level)
        
        return features
    
    def _calculate_equipment_value_advantage(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算装备价值优势"""
        # 简化实现：基于总经济的装备价值估算
        my_total = float(my_hero.get("moneyCnt", 0))
        enemy_total = float(enemy_hero.get("moneyCnt", 0))
        
        # 假设70%的总经济转化为装备价值
        my_equipment_value = my_total * 0.7
        enemy_equipment_value = enemy_total * 0.7
        
        advantage = (my_equipment_value - enemy_equipment_value) / 10000.0
        return max(-1.0, min(1.0, advantage))
    
    def _calculate_last_hit_potential(self, my_hero: Dict, frame_state: Dict) -> float:
        """计算补刀潜力"""
        my_pos = self._get_position(my_hero)
        npcs = frame_state.get('npc_states', [])
        
        low_hp_minions = 0
        for npc in npcs:
            if 'SOLDIER' in npc.get('sub_type', ''):
                npc_hp_ratio = float(npc.get('hp', 0)) / max(float(npc.get('max_hp', 1)), 1)
                if npc_hp_ratio < 0.3:  # 低血量小兵
                    npc_pos = npc.get('location', {})
                    if npc_pos:
                        dist = self._calculate_distance_by_pos(my_pos, npc_pos)
                        if dist <= 1200:  # 攻击范围内
                            low_hp_minions += 1
        
        return min(low_hp_minions / 3.0, 1.0)  # 归一化到[0,1]
    
    def _calculate_catchup_potential(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> float:
        """计算经济追赶潜力"""
        my_money = float(my_hero.get("money", 0))
        enemy_money = float(enemy_hero.get("money", 0))
        
        if my_money >= enemy_money:
            return 0.0  # 已经领先，无需追赶
        
        # 基于小兵数量和补刀机会评估追赶潜力
        minion_opportunity = self._calculate_last_hit_potential(my_hero, frame_state)
        money_gap = (enemy_money - my_money) / 5000.0  # 归一化差距
        
        # 差距越小，小兵机会越多，追赶潜力越大
        catchup_potential = minion_opportunity * (1.0 - min(money_gap, 1.0))
        return catchup_potential
    
    def _evaluate_minion_last_hit_opportunity(self, my_hero: Dict, frame_state: Dict) -> float:
        """评估小兵补刀机会"""
        return self._calculate_last_hit_potential(my_hero, frame_state)
    
    def _evaluate_safe_farming_state(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> float:
        """评估安全发育状态"""
        my_hp_ratio = self._get_hp_ratio(my_hero)
        distance = self._calculate_distance(my_hero, enemy_hero)
        minion_opportunity = self._calculate_last_hit_potential(my_hero, frame_state)
        
        # 血量充足 + 距离安全 + 有小兵机会
        safe_hp = 1.0 if my_hp_ratio > 0.6 else 0.0
        safe_distance = 1.0 if distance > 800 else 0.0
        
        return (safe_hp + safe_distance + minion_opportunity) / 3.0
    
    def _evaluate_equipment_purchase_timing(self, my_hero: Dict) -> float:
        """评估装备购买时机"""
        current_money = float(my_hero.get("money", 0))
        
        # 简化：基于金币数量判断购买时机
        if current_money >= 3000:  # 可以买大件
            return 1.0
        elif current_money >= 1500:  # 可以买中等装备
            return 0.6
        elif current_money >= 800:  # 可以买小件
            return 0.3
        else:
            return 0.0
    
    def _evaluate_economic_risk(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> float:
        """评估经济风险等级"""
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        distance = self._calculate_distance(my_hero, enemy_hero)
        
        # 风险因素
        low_hp_risk = 1.0 if my_hp_ratio < 0.4 else 0.0
        enemy_advantage_risk = 1.0 if enemy_hp_ratio > my_hp_ratio + 0.3 else 0.0
        close_distance_risk = 1.0 if distance < 600 else 0.0
        
        risk_level = (low_hp_risk + enemy_advantage_risk + close_distance_risk) / 3.0
        return risk_level
    
    def _extract_interaction_features(self, my_hero: Dict, frame_state: Dict) -> List[float]:
        """提取环境交互特征"""
        features = []
        
        # 与小兵的交互
        minion_interaction = self._analyze_minion_interaction(my_hero, frame_state)
        features.extend(minion_interaction[:5])
        
        # 与防御塔的交互
        tower_interaction = self._analyze_tower_interaction(my_hero, frame_state)
        features.extend(tower_interaction[:3])
        
        # 与野怪的交互
        jungle_interaction = self._analyze_jungle_interaction(my_hero, frame_state)
        features.extend(jungle_interaction[:2])
        
        return features
    
    def _analyze_minion_interaction(self, my_hero: Dict, frame_state: Dict) -> List[float]:
        """分析与小兵的交互"""
        features = []
        
        npcs = frame_state.get("npc_states", [])
        my_pos = self._get_position(my_hero)
        
        # 统计附近小兵
        nearby_ally_minions = 0
        nearby_enemy_minions = 0
        
        for npc in npcs:
            if "SOLDIER" in npc.get("sub_type", ""):
                npc_pos = npc.get("location", {})
                dist = self._calculate_distance_by_pos(my_pos, npc_pos)
                
                if dist <= 3000:  # 3000范围内
                    if npc.get("camp") == self.main_camp:
                        nearby_ally_minions += 1
                    else:
                        nearby_enemy_minions += 1
        
        features.extend([
            min(nearby_ally_minions, 5) / 5.0,
            min(nearby_enemy_minions, 5) / 5.0,
            0.0, 0.0, 0.0  # 预留位置
        ])
        
        return features
    
    def _analyze_tower_interaction(self, my_hero: Dict, frame_state: Dict) -> List[float]:
        """分析与防御塔的交互"""
        features = []
        
        npcs = frame_state.get("npc_states", [])
        my_pos = self._get_position(my_hero)
        
        # 查找最近的敌方防御塔
        min_enemy_tower_dist = float('inf')
        in_enemy_tower_range = 0.0
        
        for npc in npcs:
            if "TOWER" in npc.get("sub_type", "") and npc.get("camp") != self.main_camp:
                tower_pos = npc.get("location", {})
                dist = self._calculate_distance_by_pos(my_pos, tower_pos)
                min_enemy_tower_dist = min(min_enemy_tower_dist, dist)
                
                # 检查是否在塔的攻击范围内
                tower_range = npc.get("attack_range", 0)
                if dist <= tower_range:
                    in_enemy_tower_range = 1.0
        
        features.extend([
            min(min_enemy_tower_dist, 15000.0) / 15000.0,
            in_enemy_tower_range,
            0.0  # 预留
        ])
        
        return features
    
    def _analyze_jungle_interaction(self, my_hero: Dict, frame_state: Dict) -> List[float]:
        """分析与野怪的交互"""
        # 简化实现，预留扩展
        return [0.0, 0.0]
    
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
    
    def _get_position(self, hero: Dict) -> Dict:
        """获取英雄位置"""
        if not isinstance(hero, dict):
            return {"x": 0, "z": 0}
        
        actor_state = hero.get("actor_state", {})
        if not isinstance(actor_state, dict):
            return {"x": 0, "z": 0}
        
        location = actor_state.get("location", {})
        if not isinstance(location, dict):
            return {"x": 0, "z": 0}
        
        return location
    
    def _calculate_distance(self, hero1: Dict, hero2: Dict) -> float:
        """计算两个英雄之间的距离"""
        pos1 = self._get_position(hero1)
        pos2 = self._get_position(hero2)
        return self._calculate_distance_by_pos(pos1, pos2)
    
    def _calculate_distance_by_pos(self, pos1: Dict, pos2: Dict) -> float:
        """根据位置计算距离"""
        x1, z1 = float(pos1.get("x", 0)), float(pos1.get("z", 0))
        x2, z2 = float(pos2.get("x", 0)), float(pos2.get("z", 0))
        return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


class SunShangxiangComboDetector:
    """孙尚香连招检测器"""
    
    def __init__(self):
        self.skill_history = deque(maxlen=30)  # 技能使用历史
        self.combo_windows = {
            's2_mark_window': -1,      # S2标记窗口
            's1_enhance_window': -1,   # S1强化普攻窗口
            'combo_execution_window': -1  # 连招执行窗口
        }
    
    def extract_combo_features(self, hero: Dict, frame_no: int, legal_actions: List) -> List[float]:
        """提取连招相关特征"""
        features = []
        
        # 技能状态特征
        skill_states = self._get_skill_states(hero)
        features.extend(skill_states[:12])  # 12维技能状态
        
        # 连招窗口状态
        combo_windows = self._get_combo_windows(frame_no)
        features.extend(combo_windows[:6])   # 6维连招窗口
        
        # 连招机会评估
        combo_opportunities = self._evaluate_combo_opportunities(hero, legal_actions)
        features.extend(combo_opportunities[:8])  # 8维连招机会
        
        # 技能命中率统计
        hit_rates = self._get_skill_hit_rates(hero)
        features.extend(hit_rates[:4])  # 4维命中率
        
        return features
    
    def _get_skill_states(self, hero: Dict) -> List[float]:
        """获取技能状态"""
        features = []
        skill_state = hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        
        for i in range(4):  # 4个技能槽
            if i < len(slots):
                slot = slots[i]
                features.extend([
                    float(slot.get("usable", 0)),
                    float(slot.get("cooldown", 0)) / max(slot.get("cooldown_max", 1), 1),
                    float(slot.get("level", 0)) / 15.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _get_combo_windows(self, frame_no: int) -> List[float]:
        """获取连招窗口状态"""
        features = []
        
        for window_name, end_frame in self.combo_windows.items():
            if end_frame > frame_no:
                remaining = (end_frame - frame_no) / 60.0  # 归一化到秒
                features.append(min(remaining, 1.0))
            else:
                features.append(0.0)
        
        # 填充到6维
        while len(features) < 6:
            features.append(0.0)
        
        return features
    
    def _evaluate_combo_opportunities(self, hero: Dict, legal_actions: List) -> List[float]:
        """评估连招机会"""
        # 简化实现，基于技能可用性和合法动作
        features = []
        
        # S2可用且合法
        s2_available = self._is_skill_available_and_legal(hero, 1, legal_actions)
        features.append(float(s2_available))
        
        # S1可用且合法
        s1_available = self._is_skill_available_and_legal(hero, 2, legal_actions)
        features.append(float(s1_available))
        
        # 普攻可用
        aa_available = len(legal_actions) > 3 and legal_actions[3] > 0
        features.append(float(aa_available))
        
        # 填充到8维
        while len(features) < 8:
            features.append(0.0)
        
        return features
    
    def _get_skill_hit_rates(self, hero: Dict) -> List[float]:
        """获取技能命中率"""
        features = []
        skill_state = hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        
        for i in range(4):
            if i < len(slots):
                slot = slots[i]
                used_times = float(slot.get("usedTimes", 0))
                hit_times = float(slot.get("hitHeroTimes", 0))
                hit_rate = hit_times / max(used_times, 1.0)
                features.append(hit_rate)
            else:
                features.append(0.0)
        
        return features
    
    def _is_skill_available_and_legal(self, hero: Dict, skill_index: int, legal_actions: List) -> bool:
        """检查技能是否可用且合法"""
        skill_state = hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        
        if skill_index >= len(slots):
            return False
        
        slot = slots[skill_index]
        usable = slot.get("usable", False)
        
        # 检查合法动作
        legal = len(legal_actions) > skill_index + 3 and legal_actions[skill_index + 3] > 0
        
        return usable and legal


class MarksmanTacticalAnalyzer:
    """射手战术分析器"""
    
    def extract_tactical_features(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """提取射手战术特征"""
        features = []
        
        # 风筝战术特征 (8维)
        kiting_features = self._analyze_kiting_potential(my_hero, enemy_hero)
        features.extend(kiting_features)
        
        # 位置优势特征 (8维)
        position_features = self._analyze_position_advantage(my_hero, enemy_hero, frame_state)
        features.extend(position_features)
        
        # 输出窗口特征 (5维)
        damage_window_features = self._analyze_damage_windows(my_hero, enemy_hero)
        features.extend(damage_window_features)
        
        # 安全评估特征 (4维)
        safety_features = self._analyze_safety_level(my_hero, enemy_hero, frame_state)
        features.extend(safety_features)
        
        return features
    
    def _analyze_kiting_potential(self, my_hero: Dict, enemy_hero: Dict) -> List[float]:
        """分析风筝潜力"""
        features = []
        
        # 速度优势
        my_speed = self._safe_get_nested(my_hero, ["actor_state", "values", "mov_spd"], 0)
        enemy_speed = self._safe_get_nested(enemy_hero, ["actor_state", "values", "mov_spd"], 0)
        speed_advantage = (my_speed - enemy_speed) / 1000.0
        features.append(speed_advantage)
        
        # 攻击范围优势
        my_range = self._safe_get_nested(my_hero, ["actor_state", "attack_range"], 0)
        enemy_range = self._safe_get_nested(enemy_hero, ["actor_state", "attack_range"], 0)
        range_advantage = (my_range - enemy_range) / 1000.0
        features.append(range_advantage)
        
        # 填充到8维
        while len(features) < 8:
            features.append(0.0)
        
        return features
    
    def _analyze_position_advantage(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """分析位置优势"""
        features = []
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 相对位置
        dx = float(my_pos.get("x", 0)) - float(enemy_pos.get("x", 0))
        dz = float(my_pos.get("z", 0)) - float(enemy_pos.get("z", 0))
        
        features.extend([
            dx / 15000.0,  # 归一化x差值
            dz / 15000.0,  # 归一化z差值
        ])
        
        # 填充到8维
        while len(features) < 8:
            features.append(0.0)
        
        return features
    
    def _analyze_damage_windows(self, my_hero: Dict, enemy_hero: Dict) -> List[float]:
        """分析输出窗口"""
        features = []
        
        # 血量差异
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        hp_advantage = my_hp_ratio - enemy_hp_ratio
        features.append(hp_advantage)
        
        # 填充到5维
        while len(features) < 5:
            features.append(0.0)
        
        return features
    
    def _analyze_safety_level(self, my_hero: Dict, enemy_hero: Dict, frame_state: Dict) -> List[float]:
        """分析安全等级"""
        features = []
        
        # 血量安全度
        my_hp_ratio = self._get_hp_ratio(my_hero)
        features.append(my_hp_ratio)
        
        # 填充到4维
        while len(features) < 4:
            features.append(0.0)
        
        return features
    
    def _get_hp_ratio(self, hero: Dict) -> float:
        """获取血量比例"""
        actor_state = hero.get("actor_state", {})
        hp = float(actor_state.get("hp", 0))
        max_hp = float(actor_state.get("max_hp", 1))
        return hp / max(max_hp, 1.0)


class StateTracker:
    """状态追踪器"""
    
    def __init__(self):
        self.history = deque(maxlen=10)
    
    def extract_temporal_features(self, my_hero: Dict, enemy_hero: Dict, frame_no: int) -> List[float]:
        """提取时序动态特征"""
        current_state = {
            'frame_no': frame_no,
            'my_hp': self._get_hp_ratio(my_hero),
            'enemy_hp': self._get_hp_ratio(enemy_hero),
            'distance': self._calculate_distance(my_hero, enemy_hero)
        }
        
        self.history.append(current_state)
        
        features = []
        
        if len(self.history) >= 2:
            prev_state = self.history[-2]
            
            # 血量变化趋势
            my_hp_change = current_state['my_hp'] - prev_state['my_hp']
            enemy_hp_change = current_state['enemy_hp'] - prev_state['enemy_hp']
            features.extend([my_hp_change, enemy_hp_change])
            
            # 距离变化趋势
            distance_change = current_state['distance'] - prev_state['distance']
            features.append(distance_change / 1000.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 填充到15维
        while len(features) < 15:
            features.append(0.0)
        
        return features
    
    def _get_hp_ratio(self, hero: Dict) -> float:
        """获取血量比例"""
        actor_state = hero.get("actor_state", {})
        hp = float(actor_state.get("hp", 0))
        max_hp = float(actor_state.get("max_hp", 1))
        return hp / max(max_hp, 1.0)
    
    def _calculate_distance(self, hero1: Dict, hero2: Dict) -> float:
        """计算距离"""
        pos1 = self._get_position(hero1)
        pos2 = self._get_position(hero2)
        
        x1, z1 = float(pos1.get("x", 0)), float(pos1.get("z", 0))
        x2, z2 = float(pos2.get("x", 0)), float(pos2.get("z", 0))
        
        return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
    
    def _safe_get_nested(self, obj: Dict, keys: List[str], default=0.0) -> float:
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
