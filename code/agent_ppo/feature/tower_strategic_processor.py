#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Enhanced by Deep RL Analysis

防御塔战略处理器
基于深度强化学习分析，实现完整的防御塔特征工程和奖励系统

核心理念：
1. 防御塔是1v1模式的唯一胜利条件
2. "发育"与"推塔"的动态权衡
3. 基于兵线优势的战略决策
4. 机会成本的量化评估
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict


class TowerStrategicProcessor:
    """防御塔战略处理器 - 实现基于防御塔的完整特征工程"""
    
    def __init__(self, camp: str):
        self.main_camp = camp
        self.enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 历史状态追踪
        self.tower_history = deque(maxlen=50)
        self.minion_wave_history = deque(maxlen=20)
        
        # 游戏常数配置
        self.game_constants = {
            'minion_gold_value': 20,        # 小兵平均金币价值
            'cannon_minion_gold': 60,       # 炮车金币价值
            'tower_attack_range': 1000,     # 防御塔攻击范围
            'tower_attack_damage': 200,     # 防御塔攻击伤害
            'minion_hp_threshold': 0.3,     # 可补刀血量阈值
            'safe_tower_hp_threshold': 0.4, # 防御塔安全血量阈值
        }
        
    def extract_tower_strategic_features(self, observation: Dict, frame_no: int) -> List[float]:
        """提取防御塔战略特征 - 针对拆塔胜利条件优化 (35维)"""
        if not isinstance(observation, dict):
            return [0.0] * 35
        
        frame_state = observation.get("frame_state", {})
        if not isinstance(frame_state, dict):
            return [0.0] * 35
        
        # 获取英雄和防御塔状态
        hero_states = frame_state.get("hero_states", [])
        npc_states = frame_state.get("npc_states", [])
        
        my_hero = self._find_hero_by_camp(hero_states, self.main_camp)
        enemy_hero = self._find_hero_by_camp(hero_states, self.enemy_camp)
        tower_states = self._extract_tower_states(npc_states)
        minion_states = self._extract_minion_states(npc_states)
        
        if not my_hero or not enemy_hero:
            return [0.0] * 35
        
        all_features = []
        
        # 1. 基础状态特征 (4维)
        basic_features = self._extract_basic_tower_features(tower_states)
        all_features.extend(basic_features)
        
        # 2. 动态交互特征 (8维)
        interaction_features = self._extract_tower_interaction_features(
            my_hero, enemy_hero, tower_states, minion_states
        )
        all_features.extend(interaction_features)
        
        # 3. 空间与相对位置特征 (6维)
        spatial_features = self._extract_spatial_features(
            my_hero, enemy_hero, tower_states, minion_states
        )
        all_features.extend(spatial_features)
        
        # 4. 兵线状态与潜力特征 (6维)
        wave_features = self._extract_wave_potential_features(
            minion_states, tower_states, my_hero, enemy_hero
        )
        all_features.extend(wave_features)
        
        # 5. 机会成本特征 (3维)
        opportunity_features = self._extract_opportunity_cost_features(
            minion_states, my_hero, enemy_hero, tower_states
        )
        all_features.extend(opportunity_features)
        
        # 6. 情景判断组合特征 (3维)
        situational_features = self._extract_situational_features(
            my_hero, enemy_hero, tower_states, minion_states
        )
        all_features.extend(situational_features)
        
        # 7. 【新增】胜利条件导向特征 (5维) - 针对拆塔胜利优化
        victory_focused_features = self._extract_victory_focused_features(
            my_hero, enemy_hero, tower_states, minion_states, frame_no
        )
        all_features.extend(victory_focused_features)
        
        # 更新历史状态
        self._update_tower_history(tower_states, minion_states, frame_no)
        
        return all_features[:35]  # 确保返回35维特征
    
    def _extract_basic_tower_features(self, tower_states: Dict) -> List[float]:
        """提取基础防御塔状态特征 (4维)"""
        features = []
        
        # 我方防御塔血量百分比
        my_tower_hp_ratio = tower_states.get('my_tower_hp_ratio', 1.0)
        features.append(my_tower_hp_ratio)
        
        # 敌方防御塔血量百分比
        enemy_tower_hp_ratio = tower_states.get('enemy_tower_hp_ratio', 1.0)
        features.append(enemy_tower_hp_ratio)
        
        # 防御塔血量差 (势能函数的基础)
        tower_hp_advantage = my_tower_hp_ratio - enemy_tower_hp_ratio
        features.append(tower_hp_advantage)
        
        # 双方防御塔健康度乘积 (反映游戏进程)
        tower_hp_product = my_tower_hp_ratio * enemy_tower_hp_ratio
        features.append(tower_hp_product)
        
        return features
    
    def _extract_tower_interaction_features(self, my_hero: Dict, enemy_hero: Dict, 
                                          tower_states: Dict, minion_states: Dict) -> List[float]:
        """提取动态交互特征 (8维)"""
        features = []
        
        # 我方防御塔是否正被攻击
        my_tower_under_attack = 1.0 if tower_states.get('my_tower_under_attack', False) else 0.0
        features.append(my_tower_under_attack)
        
        # 攻击我方塔的敌方小兵数量
        enemy_minions_attacking_my_tower = float(tower_states.get('enemy_minions_attacking_my_tower', 0))
        features.append(min(enemy_minions_attacking_my_tower / 5.0, 1.0))  # 归一化
        
        # 敌方英雄是否正在攻击我方塔
        enemy_hero_attacking_my_tower = 1.0 if tower_states.get('enemy_hero_attacking_my_tower', False) else 0.0
        features.append(enemy_hero_attacking_my_tower)
        
        # 我方英雄是否在敌方塔攻击范围内
        my_hero_in_enemy_tower_range = self._is_hero_in_tower_range(my_hero, tower_states.get('enemy_tower_pos'))
        features.append(1.0 if my_hero_in_enemy_tower_range else 0.0)
        
        # 敌方防御塔是否正被我方单位攻击
        enemy_tower_under_attack = 1.0 if tower_states.get('enemy_tower_under_attack', False) else 0.0
        features.append(enemy_tower_under_attack)
        
        # 攻击敌方塔的我方小兵数量
        my_minions_attacking_enemy_tower = float(tower_states.get('my_minions_attacking_enemy_tower', 0))
        features.append(min(my_minions_attacking_enemy_tower / 5.0, 1.0))  # 归一化
        
        # 我方英雄是否正在攻击敌方塔
        my_hero_attacking_enemy_tower = self._is_hero_attacking_tower(my_hero, tower_states.get('enemy_tower_pos'))
        features.append(1.0 if my_hero_attacking_enemy_tower else 0.0)
        
        # 防御塔威胁等级 (综合指标)
        tower_threat_level = self._calculate_tower_threat_level(
            my_tower_under_attack, enemy_minions_attacking_my_tower, enemy_hero_attacking_my_tower
        )
        features.append(tower_threat_level)
        
        return features
    
    def _extract_spatial_features(self, my_hero: Dict, enemy_hero: Dict, 
                                tower_states: Dict, minion_states: Dict) -> List[float]:
        """提取空间与相对位置特征 (6维)"""
        features = []
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 我方英雄到我方防御塔的距离
        my_tower_pos = tower_states.get('my_tower_pos', (0, 0))
        dist_to_my_tower = self._calculate_distance_by_pos(my_pos, my_tower_pos)
        features.append(min(dist_to_my_tower / 3000.0, 1.0))  # 归一化
        
        # 我方英雄到敌方防御塔的距离
        enemy_tower_pos = tower_states.get('enemy_tower_pos', (0, 0))
        dist_to_enemy_tower = self._calculate_distance_by_pos(my_pos, enemy_tower_pos)
        features.append(min(dist_to_enemy_tower / 3000.0, 1.0))  # 归一化
        
        # 敌方英雄到我方防御塔的距离
        enemy_dist_to_my_tower = self._calculate_distance_by_pos(enemy_pos, my_tower_pos)
        features.append(min(enemy_dist_to_my_tower / 3000.0, 1.0))  # 归一化
        
        # 兵线交锋位置 (-1: 我方塔下, 0: 中线, +1: 敌方塔下)
        battle_line_position = self._calculate_battle_line_position(minion_states, my_tower_pos, enemy_tower_pos)
        features.append(battle_line_position)
        
        # 我方英雄相对于兵线的位置
        hero_relative_to_battle_line = self._calculate_hero_battle_line_relation(my_pos, minion_states)
        features.append(hero_relative_to_battle_line)
        
        # 地图控制权 (基于双方英雄和兵线的位置)
        map_control = self._calculate_map_control(my_hero, enemy_hero, minion_states, tower_states)
        features.append(map_control)
        
        return features
    
    def _extract_wave_potential_features(self, minion_states: Dict, tower_states: Dict, 
                                       my_hero: Dict, enemy_hero: Dict) -> List[float]:
        """提取兵线状态与潜力特征 (6维)"""
        features = []
        
        # 我方兵线进攻潜力
        my_wave_offensive_potential = self._calculate_wave_offensive_potential(
            minion_states.get('my_minions', []), tower_states.get('enemy_tower_pos', (0, 0))
        )
        features.append(my_wave_offensive_potential)
        
        # 敌方兵线防守压力
        enemy_wave_defensive_pressure = self._calculate_wave_defensive_pressure(
            minion_states.get('enemy_minions', []), tower_states.get('my_tower_pos', (0, 0))
        )
        features.append(enemy_wave_defensive_pressure)
        
        # 兵线优势度 (核心特征)
        wave_advantage = my_wave_offensive_potential - enemy_wave_defensive_pressure
        features.append(wave_advantage)
        
        # 我方兵线总血量 (健康度)
        my_wave_health = self._calculate_wave_health(minion_states.get('my_minions', []))
        features.append(my_wave_health)
        
        # 敌方兵线总血量
        enemy_wave_health = self._calculate_wave_health(minion_states.get('enemy_minions', []))
        features.append(enemy_wave_health)
        
        # 兵线血量优势
        wave_health_advantage = my_wave_health - enemy_wave_health
        features.append(wave_health_advantage)
        
        return features
    
    def _extract_opportunity_cost_features(self, minion_states: Dict, my_hero: Dict, 
                                         enemy_hero: Dict, tower_states: Dict) -> List[float]:
        """提取机会成本特征 (3维)"""
        features = []
        
        # 可获取的金币总值
        available_gold = self._calculate_available_gold(minion_states.get('enemy_minions', []), my_hero)
        features.append(min(available_gold / 300.0, 1.0))  # 归一化 (假设最大5个小兵*60金币)
        
        # 进攻机会损失
        lost_pushing_opportunity = self._calculate_lost_pushing_opportunity(
            minion_states, my_hero, tower_states
        )
        features.append(lost_pushing_opportunity)
        
        # 防守紧迫度
        defensive_urgency = self._calculate_defensive_urgency(
            minion_states, tower_states, enemy_hero
        )
        features.append(defensive_urgency)
        
        return features
    
    def _extract_situational_judgment_features(self, my_hero: Dict, enemy_hero: Dict, 
                                             tower_states: Dict, minion_states: Dict, 
                                             wave_features: List[float]) -> List[float]:
        """提取情景判断组合特征 (3维)"""
        features = []
        
        # "安全推塔"指数
        safe_pushing_index = self._calculate_safe_pushing_index(
            my_hero, enemy_hero, tower_states, minion_states, wave_features
        )
        features.append(safe_pushing_index)
        
        # "极限守塔"指数
        desperate_defense_index = self._calculate_desperate_defense_index(
            tower_states, minion_states, wave_features
        )
        features.append(desperate_defense_index)
        
        # "节奏控制"指数
        tempo_control_index = self._calculate_tempo_control_index(
            my_hero, enemy_hero, tower_states, minion_states
        )
        features.append(tempo_control_index)
        
        return features
    
    # ============ 核心计算方法 ============
    
    def _calculate_wave_offensive_potential(self, my_minions: List[Dict], enemy_tower_pos: Tuple[float, float]) -> float:
        """计算我方兵线进攻潜力"""
        if not my_minions:
            return 0.0
        
        total_potential = 0.0
        for minion in my_minions:
            minion_hp = float(minion.get('hp', 0))
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance_to_enemy_tower = self._calculate_distance_by_pos(minion_position, enemy_tower_pos)
                
                # 潜力 = 血量 / 距离，距离越近潜力越大
                if distance_to_enemy_tower > 0:
                    potential = minion_hp / max(distance_to_enemy_tower, 100.0)
                    total_potential += potential
        
        # 归一化处理
        return min(total_potential / 1000.0, 1.0)
    
    def _calculate_wave_defensive_pressure(self, enemy_minions: List[Dict], my_tower_pos: Tuple[float, float]) -> float:
        """计算敌方兵线防守压力"""
        if not enemy_minions:
            return 0.0
        
        total_pressure = 0.0
        for minion in enemy_minions:
            minion_hp = float(minion.get('hp', 0))
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance_to_my_tower = self._calculate_distance_by_pos(minion_position, my_tower_pos)
                
                # 压力 = 血量 / 距离，距离越近压力越大
                if distance_to_my_tower > 0:
                    pressure = minion_hp / max(distance_to_my_tower, 100.0)
                    total_pressure += pressure
        
        # 归一化处理
        return min(total_pressure / 1000.0, 1.0)
    
    def _calculate_wave_health(self, minions: List[Dict]) -> float:
        """计算兵线总血量"""
        total_hp = sum(float(minion.get('hp', 0)) for minion in minions)
        # 归一化：假设一波兵最大血量约3000
        return min(total_hp / 3000.0, 1.0)
    
    def _calculate_available_gold(self, enemy_minions: List[Dict], my_hero: Dict) -> float:
        """计算可获取的金币总值"""
        my_pos = self._get_position(my_hero)
        total_gold = 0.0
        
        for minion in enemy_minions:
            minion_hp_ratio = float(minion.get('hp', 0)) / max(float(minion.get('max_hp', 1)), 1)
            
            # 只计算可补刀的小兵 (低血量)
            if minion_hp_ratio < self.game_constants['minion_hp_threshold']:
                minion_pos = minion.get('location', {})
                if minion_pos:
                    minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                    distance = self._calculate_distance_by_pos(my_pos, minion_position)
                    
                    # 在攻击范围内的小兵才算可获取
                    if distance <= 1200:  # 攻击+移动范围
                        minion_type = minion.get('sub_type', '')
                        if 'CANNON' in minion_type:
                            total_gold += self.game_constants['cannon_minion_gold']
                        else:
                            total_gold += self.game_constants['minion_gold_value']
        
        return total_gold
    
    def _calculate_lost_pushing_opportunity(self, minion_states: Dict, my_hero: Dict, tower_states: Dict) -> float:
        """计算进攻机会损失"""
        # 检查是否有我方兵线在敌方塔下，但我方英雄不在场
        my_minions = minion_states.get('my_minions', [])
        enemy_tower_pos = tower_states.get('enemy_tower_pos', (0, 0))
        my_pos = self._get_position(my_hero)
        
        minions_at_enemy_tower = 0
        for minion in my_minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance_to_tower = self._calculate_distance_by_pos(minion_position, enemy_tower_pos)
                if distance_to_tower <= 1200:  # 在塔附近
                    minions_at_enemy_tower += 1
        
        # 如果有兵在敌方塔下，但我方英雄距离很远
        hero_distance_to_enemy_tower = self._calculate_distance_by_pos(my_pos, enemy_tower_pos)
        
        if minions_at_enemy_tower > 0 and hero_distance_to_enemy_tower > 2000:
            return min(minions_at_enemy_tower / 3.0, 1.0)  # 归一化
        
        return 0.0
    
    def _calculate_defensive_urgency(self, minion_states: Dict, tower_states: Dict, enemy_hero: Dict) -> float:
        """计算防守紧迫度"""
        my_tower_hp_ratio = tower_states.get('my_tower_hp_ratio', 1.0)
        enemy_minions_attacking = tower_states.get('enemy_minions_attacking_my_tower', 0)
        enemy_hero_attacking = tower_states.get('enemy_hero_attacking_my_tower', False)
        
        urgency = 0.0
        
        # 塔血量低增加紧迫度
        if my_tower_hp_ratio < 0.5:
            urgency += (0.5 - my_tower_hp_ratio) * 2.0
        
        # 敌方小兵攻击增加紧迫度
        urgency += enemy_minions_attacking * 0.2
        
        # 敌方英雄攻击大幅增加紧迫度
        if enemy_hero_attacking:
            urgency += 0.5
        
        return min(urgency, 1.0)
    
    def _calculate_safe_pushing_index(self, my_hero: Dict, enemy_hero: Dict, tower_states: Dict, 
                                    minion_states: Dict, wave_features: List[float]) -> float:
        """计算"安全推塔"指数"""
        # 基础条件检查
        wave_advantage = wave_features[2] if len(wave_features) > 2 else 0.0  # 兵线优势度
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        enemy_tower_pos = tower_states.get('enemy_tower_pos', (0, 0))
        
        # 距离敌方塔的距离
        hero_distance_to_enemy_tower = self._calculate_distance_by_pos(my_pos, enemy_tower_pos)
        enemy_distance_to_enemy_tower = self._calculate_distance_by_pos(enemy_pos, enemy_tower_pos)
        
        # 计算安全推塔指数
        safe_index = 0.0
        
        # 兵线优势
        if wave_advantage > 0.3:
            safe_index += 0.4
        
        # 我方英雄在敌方塔附近
        if hero_distance_to_enemy_tower < 1500:
            safe_index += 0.2
        
        # 敌方英雄距离较远或血量较低
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        if enemy_distance_to_enemy_tower > 2000 or enemy_hp_ratio < 0.4:
            safe_index += 0.3
        
        # 我方技能状态良好 (简化检查)
        my_hp_ratio = self._get_hp_ratio(my_hero)
        if my_hp_ratio > 0.6:
            safe_index += 0.1
        
        return min(safe_index, 1.0)
    
    def _calculate_desperate_defense_index(self, tower_states: Dict, minion_states: Dict, wave_features: List[float]) -> float:
        """计算"极限守塔"指数"""
        my_tower_hp_ratio = tower_states.get('my_tower_hp_ratio', 1.0)
        enemy_wave_pressure = wave_features[1] if len(wave_features) > 1 else 0.0  # 敌方兵线压力
        
        desperate_index = 0.0
        
        # 我方塔血量危险
        if my_tower_hp_ratio < 0.5:
            desperate_index += (0.5 - my_tower_hp_ratio) * 2.0
        
        # 敌方兵线压力大
        if enemy_wave_pressure > 0.5:
            desperate_index += enemy_wave_pressure
        
        return min(desperate_index, 1.0)
    
    def _calculate_tempo_control_index(self, my_hero: Dict, enemy_hero: Dict, 
                                     tower_states: Dict, minion_states: Dict) -> float:
        """计算"节奏控制"指数"""
        # 基于双方塔血量、英雄状态、兵线控制的综合评估
        my_tower_hp = tower_states.get('my_tower_hp_ratio', 1.0)
        enemy_tower_hp = tower_states.get('enemy_tower_hp_ratio', 1.0)
        
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        
        # 节奏控制 = 我方优势的综合体现
        tempo_index = 0.0
        
        # 塔血量优势
        tempo_index += (my_tower_hp - enemy_tower_hp) * 0.3
        
        # 英雄血量优势
        tempo_index += (my_hp_ratio - enemy_hp_ratio) * 0.2
        
        # 兵线控制 (简化评估)
        my_minions_count = len(minion_states.get('my_minions', []))
        enemy_minions_count = len(minion_states.get('enemy_minions', []))
        minion_advantage = (my_minions_count - enemy_minions_count) / 10.0
        tempo_index += minion_advantage * 0.2
        
        return max(-1.0, min(tempo_index, 1.0))  # 限制在[-1,1]范围
    
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
    
    def _get_position(self, unit: Dict) -> Tuple[float, float]:
        """获取单位位置（英雄或小兵）"""
        if not isinstance(unit, dict):
            return (0.0, 0.0)
        
        # 尝试从actor_state获取位置（英雄）
        actor_state = unit.get("actor_state", {})
        if isinstance(actor_state, dict) and actor_state:
            location = actor_state.get("location", {})
            if isinstance(location, dict) and location:
                x = location.get("x", 0)
                z = location.get("z", 0)
                if x is not None and z is not None:
                    return (float(x), float(z))
        
        # 尝试直接从单位获取位置（小兵或其他单位）
        location = unit.get("location", {})
        if isinstance(location, dict) and location:
            x = location.get("x", 0)
            z = location.get("z", 0)
            if x is not None and z is not None:
                return (float(x), float(z))
        
        # 尝试从其他可能的字段获取位置
        x = unit.get("x", unit.get("pos_x", 0))
        z = unit.get("z", unit.get("pos_z", unit.get("y", unit.get("pos_y", 0))))
        
        return (float(x) if x is not None else 0.0, float(z) if z is not None else 0.0)
    
    def _get_hp_ratio(self, hero: Dict) -> float:
        """获取血量比例"""
        if not isinstance(hero, dict):
            return 0.0
        
        actor_state = hero.get("actor_state", {})
        if not isinstance(actor_state, dict):
            return 0.0
        
        hp = actor_state.get("hp", 0)
        max_hp = actor_state.get("max_hp", 1)
        
        try:
            hp = float(hp) if hp is not None else 0.0
            max_hp = float(max_hp) if max_hp is not None else 1.0
            return hp / max(max_hp, 1.0)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_distance_by_pos(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点之间的距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_distance_pos(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两个位置之间的距离（别名方法）"""
        return self._calculate_distance_by_pos(pos1, pos2)
    
    def _count_minions_at_tower(self, minions: List[Dict], tower_pos: Tuple[float, float]) -> int:
        """统计在防御塔附近的小兵数量"""
        if not tower_pos or not minions:
            return 0
        
        count = 0
        tower_range = 1200  # 防御塔附近的范围
        
        for minion in minions:
            minion_pos = self._get_position(minion)
            if minion_pos:
                distance = self._calculate_distance_pos(minion_pos, tower_pos)
                if distance <= tower_range:
                    count += 1
        
        return count
    
    def _extract_tower_states(self, npcs: List) -> Dict:
        """提取防御塔状态"""
        tower_states = {
            'my_tower_hp_ratio': 1.0,
            'enemy_tower_hp_ratio': 1.0,
            'my_tower_pos': (0, 0),
            'enemy_tower_pos': (0, 0),
            'my_tower_under_attack': False,
            'enemy_tower_under_attack': False,
            'enemy_minions_attacking_my_tower': 0,
            'my_minions_attacking_enemy_tower': 0,
            'enemy_hero_attacking_my_tower': False,
        }
        
        if isinstance(npcs, list):
            for npc in npcs:
                if not isinstance(npc, dict):
                    continue
                
                if 'TOWER' in npc.get('sub_type', ''):
                    try:
                        hp = float(npc.get('hp', 0))
                        max_hp = float(npc.get('max_hp', 1))
                        hp_ratio = hp / max(max_hp, 1)
                        
                        npc_pos = npc.get('location', {})
                        if isinstance(npc_pos, dict):
                            x = float(npc_pos.get('x', 0))
                            z = float(npc_pos.get('z', 0))
                            position = (x, z)
                        else:
                            position = (0.0, 0.0)
                        
                        if npc.get('camp') == self.main_camp:
                            tower_states['my_tower_hp_ratio'] = hp_ratio
                            tower_states['my_tower_pos'] = position
                        else:
                            tower_states['enemy_tower_hp_ratio'] = hp_ratio
                            tower_states['enemy_tower_pos'] = position
                    except (ValueError, TypeError):
                        continue
        
        return tower_states
    
    def _extract_minion_states(self, npcs: List) -> Dict:
        """提取小兵状态"""
        minion_states = {
            'my_minions': [],
            'enemy_minions': []
        }
        
        if isinstance(npcs, list):
            for npc in npcs:
                if not isinstance(npc, dict):
                    continue
                
                if 'SOLDIER' in npc.get('sub_type', ''):
                    if npc.get('camp') == self.main_camp:
                        minion_states['my_minions'].append(npc)
                    else:
                        minion_states['enemy_minions'].append(npc)
        
        return minion_states
    
    def _is_hero_in_tower_range(self, hero: Dict, tower_pos: Tuple[float, float]) -> bool:
        """判断英雄是否在防御塔攻击范围内"""
        if not tower_pos or tower_pos == (0, 0):
            return False
        
        hero_pos = self._get_position(hero)
        distance = self._calculate_distance_by_pos(hero_pos, tower_pos)
        return distance <= self.game_constants['tower_attack_range']
    
    def _is_hero_attacking_tower(self, hero: Dict, tower_pos: Tuple[float, float]) -> bool:
        """判断英雄是否正在攻击防御塔"""
        # 简化实现：基于距离判断
        hero_pos = self._get_position(hero)
        distance = self._calculate_distance_by_pos(hero_pos, tower_pos)
        actor_state = hero.get("actor_state", {})
        if isinstance(actor_state, dict):
            attack_range = float(actor_state.get("attack_range", 600))
        else:
            attack_range = 600.0
        return distance <= attack_range + 100  # 稍微放宽范围
    
    def _calculate_tower_threat_level(self, under_attack: float, minion_count: float, hero_attacking: float) -> float:
        """计算防御塔威胁等级"""
        threat = 0.0
        if under_attack > 0:
            threat += 0.3
        threat += minion_count * 0.1
        if hero_attacking > 0:
            threat += 0.5
        return min(threat, 1.0)
    
    def _calculate_battle_line_position(self, minion_states: Dict, my_tower_pos: Tuple[float, float], 
                                      enemy_tower_pos: Tuple[float, float]) -> float:
        """计算兵线交锋位置"""
        my_minions = minion_states.get('my_minions', [])
        enemy_minions = minion_states.get('enemy_minions', [])
        
        if not my_minions and not enemy_minions:
            return 0.0
        
        # 计算兵线重心
        total_x, total_count = 0.0, 0
        
        for minion in my_minions + enemy_minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                total_x += float(minion_pos.get('x', 0))
                total_count += 1
        
        if total_count == 0:
            return 0.0
        
        battle_line_x = total_x / total_count
        
        # 将位置映射到[-1, 1]范围
        # -1: 我方塔附近, 0: 中线, +1: 敌方塔附近
        map_center_x = (my_tower_pos[0] + enemy_tower_pos[0]) / 2
        map_half_width = abs(enemy_tower_pos[0] - my_tower_pos[0]) / 2
        
        if map_half_width > 0:
            relative_position = (battle_line_x - map_center_x) / map_half_width
            return max(-1.0, min(1.0, relative_position))
        
        return 0.0
    
    def _calculate_hero_battle_line_relation(self, hero_pos: Tuple[float, float], minion_states: Dict) -> float:
        """计算英雄相对于兵线的位置"""
        # 简化实现：返回0表示中性位置
        return 0.0
    
    def _calculate_map_control(self, my_hero: Dict, enemy_hero: Dict, minion_states: Dict, tower_states: Dict) -> float:
        """计算地图控制权"""
        # 基于英雄位置、兵线分布、塔血量的综合评估
        control = 0.0
        
        # 塔血量优势
        my_tower_hp = tower_states.get('my_tower_hp_ratio', 1.0)
        enemy_tower_hp = tower_states.get('enemy_tower_hp_ratio', 1.0)
        control += (my_tower_hp - enemy_tower_hp) * 0.4
        
        # 兵线数量优势
        my_minion_count = len(minion_states.get('my_minions', []))
        enemy_minion_count = len(minion_states.get('enemy_minions', []))
        minion_advantage = (my_minion_count - enemy_minion_count) / 10.0
        control += minion_advantage * 0.3
        
        # 英雄血量优势
        my_hp = self._get_hp_ratio(my_hero)
        enemy_hp = self._get_hp_ratio(enemy_hero)
        control += (my_hp - enemy_hp) * 0.3
        
        return max(-1.0, min(control, 1.0))
    
    def _update_tower_history(self, tower_states: Dict, minion_states: Dict, frame_no: int):
        """更新防御塔历史状态"""
        history_entry = {
            'frame_no': frame_no,
            'tower_states': tower_states.copy(),
            'minion_states': {
                'my_minion_count': len(minion_states.get('my_minions', [])),
                'enemy_minion_count': len(minion_states.get('enemy_minions', []))
            }
        }
        
        self.tower_history.append(history_entry)
    
    def _extract_victory_focused_features(self, my_hero: Dict, enemy_hero: Dict, 
                                        tower_states: Dict, minion_states: Dict, frame_no: int) -> List[float]:
        """提取胜利条件导向特征 (5维) - 专门针对拆塔胜利优化"""
        features = []
        
        my_tower_hp = tower_states.get('my_tower_hp_ratio', 1.0)
        enemy_tower_hp = tower_states.get('enemy_tower_hp_ratio', 1.0)
        
        # 1. 终局紧迫度 (End Game Urgency)
        # 当任意一方塔血量很低时，游戏进入终局阶段
        my_tower_critical = 1.0 if my_tower_hp <= 0.25 else 0.0
        enemy_tower_critical = 1.0 if enemy_tower_hp <= 0.25 else 0.0
        endgame_urgency = max(my_tower_critical, enemy_tower_critical)
        features.append(endgame_urgency)
        
        # 2. 一波结束潜力 (One-Push Victory Potential)
        # 评估我方是否有能力一波推掉敌方塔
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        my_minions_at_enemy_tower = self._count_minions_at_tower(
            minion_states.get('my_minions', []), tower_states.get('enemy_tower_pos', (0, 0))
        )
        
        # 一波结束的条件：敌方塔血量低 + 我方状态好 + 有兵线支持
        one_push_potential = 0.0
        if enemy_tower_hp <= 0.4:  # 敌方塔血量低于40%
            if my_hp_ratio > 0.6:  # 我方血量充足
                one_push_potential += 0.4
            if my_minions_at_enemy_tower >= 3:  # 有足够兵线
                one_push_potential += 0.3
            if enemy_hp_ratio < 0.5:  # 敌方英雄状态不佳
                one_push_potential += 0.3
        features.append(min(one_push_potential, 1.0))
        
        # 3. 防守绝境指数 (Defensive Desperation Index)
        # 评估我方塔的危险程度，需要不惜一切代价防守
        defensive_desperation = 0.0
        if my_tower_hp <= 0.3:  # 我方塔血量低于30%
            enemy_minions_at_my_tower = self._count_minions_at_tower(
                minion_states.get('enemy_minions', []), tower_states.get('my_tower_pos', (0, 0))
            )
            
            # 基础危险度
            defensive_desperation = (0.3 - my_tower_hp) / 0.3  # 0-1范围
            
            # 如果敌方有兵线压塔，危险度增加
            if enemy_minions_at_my_tower > 0:
                defensive_desperation = min(defensive_desperation + 0.5, 1.0)
                
            # 如果敌方英雄也在附近，极度危险
            enemy_pos = self._get_position(enemy_hero)
            my_tower_pos = tower_states.get('my_tower_pos', (0, 0))
            if enemy_pos and my_tower_pos:
                distance_to_my_tower = self._calculate_distance_pos(enemy_pos, my_tower_pos)
                if distance_to_my_tower < 1500:  # 敌方英雄在塔附近
                    defensive_desperation = 1.0
        
        features.append(defensive_desperation)
        
        # 4. 塔血量差异放大器 (Tower HP Difference Amplifier)
        # 在终局阶段，塔血量差异的重要性被放大
        tower_hp_diff = my_tower_hp - enemy_tower_hp
        amplified_diff = tower_hp_diff
        
        # 在终局阶段（任意一方塔血量低于50%），差异被放大
        if min(my_tower_hp, enemy_tower_hp) < 0.5:
            amplifier = 2.0 - min(my_tower_hp, enemy_tower_hp)  # 塔血量越低，放大倍数越大
            amplified_diff = tower_hp_diff * amplifier
        
        # 归一化到[-1, 1]范围
        amplified_diff = max(-1.0, min(amplified_diff, 1.0))
        features.append(amplified_diff)
        
        # 5. 决战时机判断 (Decisive Battle Timing)
        # 判断当前是否是决定性战斗的最佳时机
        decisive_timing = 0.0
        
        # 双方塔血量都不高，进入决战阶段
        if my_tower_hp < 0.6 and enemy_tower_hp < 0.6:
            # 我方优势时，应该主动寻求决战
            if my_hp_ratio > enemy_hp_ratio and tower_hp_diff > 0:
                decisive_timing = 0.8
            # 我方劣势时，应该避免决战，寻求发育
            elif my_hp_ratio < enemy_hp_ratio and tower_hp_diff < 0:
                decisive_timing = -0.8
            # 势均力敌时，谨慎决战
            else:
                decisive_timing = 0.2
        
        features.append(decisive_timing)
        
        return features
    
    def _extract_situational_features(self, my_hero: Dict, enemy_hero: Dict, 
                                    tower_states: Dict, minion_states: Dict) -> List[float]:
        """提取情景判断特征 (3维)"""
        features = []
        
        my_tower_hp = tower_states.get('my_tower_hp_ratio', 1.0)
        enemy_tower_hp = tower_states.get('enemy_tower_hp_ratio', 1.0)
        
        # 计算兵线优势
        my_wave_potential = self._calculate_wave_offensive_potential(
            minion_states.get('my_minions', []), tower_states.get('enemy_tower_pos', (0, 0))
        )
        enemy_wave_pressure = self._calculate_wave_defensive_pressure(
            minion_states.get('enemy_minions', []), tower_states.get('my_tower_pos', (0, 0))
        )
        wave_advantage = my_wave_potential - enemy_wave_pressure
        
        # 1. "安全推塔"指数
        safe_push_index = 0.0
        if wave_advantage > 0.3:  # 兵线优势
            my_hp_ratio = self._get_hp_ratio(my_hero)
            enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
            
            if my_hp_ratio > 0.7 and enemy_hp_ratio < 0.5:  # 我方状态好，敌方状态差
                safe_push_index = 0.8
            elif my_hp_ratio > 0.5:  # 我方状态还行
                safe_push_index = 0.5
        
        features.append(safe_push_index)
        
        # 2. "极限守塔"指数
        desperate_defense_index = 0.0
        if enemy_wave_pressure > 0.5 and my_tower_hp < 0.5:
            desperate_defense_index = min(enemy_wave_pressure + (0.5 - my_tower_hp), 1.0)
        
        features.append(desperate_defense_index)
        
        # 3. "战略平衡"指数
        # 当前局势是否适合发育而非强推
        strategic_balance = 0.0
        if abs(wave_advantage) < 0.2:  # 兵线相对均衡
            my_hp_ratio = self._get_hp_ratio(my_hero)
            enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
            
            if abs(my_hp_ratio - enemy_hp_ratio) < 0.2:  # 双方状态相近
                strategic_balance = 0.8  # 适合发育
        
        features.append(strategic_balance)
        
        return features
