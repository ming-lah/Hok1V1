#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Enhanced by Deep RL Analysis

草丛战术处理器
基于深度强化学习分析，实现完整的草丛特征工程

核心理念：
1. 草丛是1v1对线中最重要的战术资源
2. 从"平面战场"到"立体空间"的维度升级
3. 视野博弈、战术欺诈和心理压迫的核心
4. 信息不对称创造、伤害规避、心理压力施加
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict


class BushTacticalProcessor:
    """草丛战术处理器 - 实现基于草丛的完整特征工程"""
    
    def __init__(self, camp: str):
        self.main_camp = camp
        self.enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 历史状态追踪
        self.vision_history = deque(maxlen=300)  # 10秒视野历史
        self.bush_activity_history = deque(maxlen=150)  # 5秒草丛活动历史
        self.enemy_position_history = deque(maxlen=90)  # 3秒敌方位置历史
        
        # 草丛状态追踪
        self.bush_state = {
            'last_enemy_seen_position': (0, 0),
            'enemy_last_seen_frame': -1,
            'in_bush_duration': 0,
            'last_ambush_frame': -1,
            'vision_pressure_duration': 0,
        }
        
        # 地图草丛定义 (基于墨家机关道地图)
        self.bush_definitions = {
            'top_bush': {        # 上方草丛 (靠近敌方塔，进攻草)
                'center': (10000, 8000),
                'radius': 800,
                'type': 'offensive'
            },
            'bottom_bush': {     # 下方草丛 (靠近我方塔，防守草)
                'center': (5000, 8000),
                'radius': 800,
                'type': 'defensive'
            }
        }
        
        # 视野计算常数
        self.vision_constants = {
            'hero_vision_range': 1200,      # 英雄视野范围
            'minion_vision_range': 600,     # 小兵视野范围
            'bush_stealth_radius': 800,     # 草丛隐身半径
            'vision_check_interval': 30,    # 视野检查间隔 (帧)
            'ambush_cooldown': 150,         # 伏击冷却时间 (5秒)
        }
        
    def extract_bush_tactical_features(self, observation: Dict, frame_no: int) -> List[float]:
        """提取草丛战术特征 (25维)"""
        if not isinstance(observation, dict):
            return [0.0] * 25
        
        frame_state = observation.get("frame_state", {})
        if not isinstance(frame_state, dict):
            return [0.0] * 25
        
        # 获取英雄和环境状态
        hero_states = frame_state.get("hero_states", [])
        npc_states = frame_state.get("npc_states", [])
        
        my_hero = self._find_hero_by_camp(hero_states, self.main_camp)
        enemy_hero = self._find_hero_by_camp(hero_states, self.enemy_camp)
        minion_states = self._extract_minion_states(npc_states)
        
        if not my_hero or not enemy_hero:
            return [0.0] * 25
        
        all_features = []
        
        # 1. 基础状态特征 (6维)
        basic_features = self._extract_basic_bush_features(my_hero, enemy_hero, minion_states)
        all_features.extend(basic_features)
        
        # 2. 视野核心特征 (7维)
        vision_features = self._extract_vision_core_features(my_hero, enemy_hero, minion_states, frame_no)
        all_features.extend(vision_features)
        
        # 3. 战略意图特征 (6维)
        strategic_features = self._extract_strategic_intent_features(my_hero, enemy_hero, minion_states)
        all_features.extend(strategic_features)
        
        # 4. 心理压迫特征 (3维)
        pressure_features = self._extract_psychological_pressure_features(my_hero, enemy_hero, frame_no)
        all_features.extend(pressure_features)
        
        # 5. 高级战术特征 (3维)
        tactical_features = self._extract_advanced_tactical_features(my_hero, enemy_hero, minion_states, frame_no)
        all_features.extend(tactical_features)
        
        # 更新历史状态
        self._update_bush_history(my_hero, enemy_hero, minion_states, frame_no)
        
        return all_features[:25]  # 确保返回25维特征
    
    def _extract_basic_bush_features(self, my_hero: Dict, enemy_hero: Dict, minion_states: Dict) -> List[float]:
        """提取基础草丛状态特征 (6维)"""
        features = []
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 我方英雄是否在草丛中
        is_self_in_bush = self._is_in_any_bush(my_pos)
        features.append(1.0 if is_self_in_bush else 0.0)
        
        # 敌方英雄是否在草丛中
        is_enemy_in_bush = self._is_in_any_bush(enemy_pos)
        features.append(1.0 if is_enemy_in_bush else 0.0)
        
        # 与最近草丛的距离
        dist_to_nearest_bush = self._distance_to_nearest_bush(my_pos)
        features.append(min(dist_to_nearest_bush / 2000.0, 1.0))  # 归一化
        
        # 敌方与最近草丛的距离
        enemy_dist_to_nearest_bush = self._distance_to_nearest_bush(enemy_pos)
        features.append(min(enemy_dist_to_nearest_bush / 2000.0, 1.0))
        
        # 我方在哪个草丛 (独热编码2维: top_bush, bottom_bush)
        my_bush_type = self._get_bush_type(my_pos)
        bush_encoding = [0.0, 0.0]
        if my_bush_type == 'top_bush':
            bush_encoding[0] = 1.0
        elif my_bush_type == 'bottom_bush':
            bush_encoding[1] = 1.0
        features.extend(bush_encoding)
        
        return features
    
    def _extract_vision_core_features(self, my_hero: Dict, enemy_hero: Dict, 
                                    minion_states: Dict, frame_no: int) -> List[float]:
        """提取视野核心特征 (7维) - 草丛战术的灵魂"""
        features = []
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 我方英雄是否对敌方可见 (核心特征)
        is_self_visible_to_enemy = self._is_visible_to_enemy(my_hero, enemy_hero, minion_states)
        features.append(1.0 if is_self_visible_to_enemy else 0.0)
        
        # 敌方英雄是否对我方可见
        is_enemy_visible_to_self = self._is_visible_to_self(enemy_hero, my_hero, minion_states)
        features.append(1.0 if is_enemy_visible_to_self else 0.0)
        
        # 敌方英雄上次可见位置的相对距离
        if self.bush_state['last_enemy_seen_position'] != (0, 0):
            last_known_pos = self.bush_state['last_enemy_seen_position']
            dist_to_last_known = self._calculate_distance_by_pos(my_pos, last_known_pos)
            features.append(min(dist_to_last_known / 3000.0, 1.0))
        else:
            features.append(0.0)
        
        # 敌方英雄消失时长
        if is_enemy_visible_to_self:
            self.bush_state['enemy_last_seen_frame'] = frame_no
            time_since_last_seen = 0
        else:
            time_since_last_seen = frame_no - self.bush_state['enemy_last_seen_frame']
        
        features.append(min(time_since_last_seen / 150.0, 1.0))  # 归一化到5秒
        
        # 视野优势指数 (我能看到敌人但敌人看不到我)
        vision_advantage = 0.0
        if is_enemy_visible_to_self and not is_self_visible_to_enemy:
            vision_advantage = 1.0
        elif not is_enemy_visible_to_self and is_self_visible_to_enemy:
            vision_advantage = -1.0
        features.append(vision_advantage)
        
        # 视野博弈强度 (双方都在草丛或都不在草丛时的特殊状态)
        vision_game_intensity = 0.0
        my_in_bush = self._is_in_any_bush(my_pos)
        enemy_in_bush = self._is_in_any_bush(enemy_pos)
        
        if my_in_bush and enemy_in_bush:
            vision_game_intensity = 1.0  # 双方都在草丛，高强度博弈
        elif not my_in_bush and not enemy_in_bush:
            vision_game_intensity = 0.2  # 双方都暴露，低强度博弈
        else:
            vision_game_intensity = 0.6  # 一方隐藏一方暴露，中等强度
        
        features.append(vision_game_intensity)
        
        # 伏击机会指数
        ambush_opportunity = self._calculate_ambush_opportunity(my_hero, enemy_hero, minion_states, frame_no)
        features.append(ambush_opportunity)
        
        return features
    
    def _extract_strategic_intent_features(self, my_hero: Dict, enemy_hero: Dict, minion_states: Dict) -> List[float]:
        """提取战略意图特征 (6维)"""
        features = []
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 草丛与战线的关系
        frontline_position = self._calculate_frontline_position(
            minion_states.get('my_minions', []), 
            minion_states.get('enemy_minions', [])
        )
        
        # 进攻草丛相对优势 (兵线靠近敌方时，进攻草丛价值更高)
        offensive_bush_value = max(0.0, frontline_position + 0.5)  # frontline > -0.5时有价值
        features.append(offensive_bush_value)
        
        # 防守草丛相对优势 (兵线靠近我方时，防守草丛价值更高)
        defensive_bush_value = max(0.0, -frontline_position + 0.5)  # frontline < 0.5时有价值
        features.append(defensive_bush_value)
        
        # 草丛的"安全度"
        top_bush_safety = self._calculate_bush_safety('top_bush', minion_states)
        bottom_bush_safety = self._calculate_bush_safety('bottom_bush', minion_states)
        
        features.extend([top_bush_safety, bottom_bush_safety])
        
        # 草丛控制权 (谁更容易控制草丛区域)
        bush_control_advantage = self._calculate_bush_control_advantage(my_hero, enemy_hero, minion_states)
        features.append(bush_control_advantage)
        
        # 草丛战术时机成熟度
        tactical_timing_readiness = self._calculate_tactical_timing(my_hero, enemy_hero)
        features.append(tactical_timing_readiness)
        
        return features
    
    def _extract_psychological_pressure_features(self, my_hero: Dict, enemy_hero: Dict, frame_no: int) -> List[float]:
        """提取心理压迫特征 (3维)"""
        features = []
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 视野压制持续时间
        is_self_in_bush = self._is_in_any_bush(my_pos)
        is_enemy_visible = self._is_visible_to_self(enemy_hero, my_hero, {})
        
        if is_self_in_bush and is_enemy_visible:
            self.bush_state['vision_pressure_duration'] += 1
        else:
            self.bush_state['vision_pressure_duration'] = 0
        
        pressure_duration_normalized = min(self.bush_state['vision_pressure_duration'] / 150.0, 1.0)  # 5秒归一化
        features.append(pressure_duration_normalized)
        
        # 敌方位置预测不确定性 (敌方消失时间越长，不确定性越高)
        time_since_seen = frame_no - self.bush_state['enemy_last_seen_frame']
        uncertainty = min(time_since_seen / 90.0, 1.0)  # 3秒后达到最大不确定性
        features.append(uncertainty)
        
        # 威慑效果强度 (基于我方在草丛中的威胁程度)
        intimidation_strength = self._calculate_intimidation_strength(my_hero, enemy_hero)
        features.append(intimidation_strength)
        
        return features
    
    def _extract_advanced_tactical_features(self, my_hero: Dict, enemy_hero: Dict, 
                                          minion_states: Dict, frame_no: int) -> List[float]:
        """提取高级战术特征 (3维)"""
        features = []
        
        # 草丛连招机会 (孙尚香特色：1技能进草 -> 强化普攻 -> 立刻回草)
        combo_opportunity = self._calculate_bush_combo_opportunity(my_hero, enemy_hero)
        features.append(combo_opportunity)
        
        # 仇恨重置价值 (当前受到多少小兵仇恨，进草能重置多少)
        aggro_reset_value = self._calculate_aggro_reset_value(my_hero, minion_states)
        features.append(aggro_reset_value)
        
        # 草丛博弈主动权 (谁在草丛博弈中占据主动)
        bush_game_initiative = self._calculate_bush_game_initiative(my_hero, enemy_hero, frame_no)
        features.append(bush_game_initiative)
        
        return features
    
    # ============ 核心计算方法 ============
    
    def _is_in_any_bush(self, position: Tuple[float, float]) -> bool:
        """判断位置是否在任何草丛中"""
        for bush_name, bush_data in self.bush_definitions.items():
            distance = self._calculate_distance_by_pos(position, bush_data['center'])
            if distance <= bush_data['radius']:
                return True
        return False
    
    def _get_bush_type(self, position: Tuple[float, float]) -> Optional[str]:
        """获取位置所在的草丛类型"""
        for bush_name, bush_data in self.bush_definitions.items():
            distance = self._calculate_distance_by_pos(position, bush_data['center'])
            if distance <= bush_data['radius']:
                return bush_name
        return None
    
    def _distance_to_nearest_bush(self, position: Tuple[float, float]) -> float:
        """计算到最近草丛的距离"""
        min_distance = float('inf')
        for bush_name, bush_data in self.bush_definitions.items():
            distance = self._calculate_distance_by_pos(position, bush_data['center'])
            # 如果在草丛内，距离为0
            if distance <= bush_data['radius']:
                return 0.0
            # 否则计算到草丛边缘的距离
            edge_distance = distance - bush_data['radius']
            min_distance = min(min_distance, edge_distance)
        
        return min_distance if min_distance != float('inf') else 2000.0
    
    def _is_visible_to_enemy(self, my_hero: Dict, enemy_hero: Dict, minion_states: Dict) -> bool:
        """判断我方英雄是否对敌方可见"""
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 如果我方不在草丛中，肯定可见
        if not self._is_in_any_bush(my_pos):
            return True
        
        # 如果我方在草丛中，检查是否有敌方单位在同一草丛内提供视野
        my_bush_type = self._get_bush_type(my_pos)
        if not my_bush_type:
            return True
        
        # 检查敌方英雄是否在同一草丛内
        if self._get_bush_type(enemy_pos) == my_bush_type:
            return True
        
        # 检查敌方小兵是否在同一草丛内
        enemy_minions = minion_states.get('enemy_minions', [])
        for minion in enemy_minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                if self._get_bush_type(minion_position) == my_bush_type:
                    return True
        
        # 在草丛中且没有敌方单位提供视野，不可见
        return False
    
    def _is_visible_to_self(self, enemy_hero: Dict, my_hero: Dict, minion_states: Dict) -> bool:
        """判断敌方英雄是否对我方可见"""
        enemy_pos = self._get_position(enemy_hero)
        my_pos = self._get_position(my_hero)
        
        # 如果敌方不在草丛中，肯定可见
        if not self._is_in_any_bush(enemy_pos):
            return True
        
        # 如果敌方在草丛中，检查是否有我方单位在同一草丛内提供视野
        enemy_bush_type = self._get_bush_type(enemy_pos)
        if not enemy_bush_type:
            return True
        
        # 检查我方英雄是否在同一草丛内
        if self._get_bush_type(my_pos) == enemy_bush_type:
            return True
        
        # 检查我方小兵是否在同一草丛内
        my_minions = minion_states.get('my_minions', [])
        for minion in my_minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                if self._get_bush_type(minion_position) == enemy_bush_type:
                    return True
        
        # 在草丛中且没有我方单位提供视野，不可见
        return False
    
    def _calculate_ambush_opportunity(self, my_hero: Dict, enemy_hero: Dict, 
                                    minion_states: Dict, frame_no: int) -> float:
        """计算伏击机会指数"""
        opportunity = 0.0
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 基础条件：我方隐藏，敌方可见
        if not self._is_visible_to_enemy(my_hero, enemy_hero, minion_states) and \
           self._is_visible_to_self(enemy_hero, my_hero, minion_states):
            opportunity += 0.4
            
            # 距离适合：在攻击范围内
            distance = self._calculate_distance_by_pos(my_pos, enemy_pos)
            actor_state = my_hero.get("actor_state", {})
            if isinstance(actor_state, dict):
                attack_range = float(actor_state.get("attack_range", 600))
            else:
                attack_range = 600.0
            if distance <= attack_range + 200:  # 稍微放宽范围
                opportunity += 0.3
            
            # 技能状态：有关键技能可用
            if self._has_key_skills_available(my_hero):
                opportunity += 0.2
            
            # 冷却检查：距离上次伏击足够长时间
            if frame_no - self.bush_state['last_ambush_frame'] > self.vision_constants['ambush_cooldown']:
                opportunity += 0.1
        
        return min(opportunity, 1.0)
    
    def _calculate_bush_safety(self, bush_name: str, minion_states: Dict) -> float:
        """计算草丛的安全度"""
        bush_data = self.bush_definitions[bush_name]
        bush_center = bush_data['center']
        
        safety = 1.0  # 基础安全度
        
        # 检查敌方小兵威胁
        enemy_minions = minion_states.get('enemy_minions', [])
        for minion in enemy_minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance = self._calculate_distance_by_pos(bush_center, minion_position)
                
                # 距离越近，安全度越低
                if distance < 1200:  # 小兵视野范围内
                    threat_factor = max(0, 1.0 - distance / 1200.0)
                    safety -= threat_factor * 0.3
        
        return max(0.0, safety)
    
    def _calculate_bush_control_advantage(self, my_hero: Dict, enemy_hero: Dict, minion_states: Dict) -> float:
        """计算草丛控制权优势"""
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        advantage = 0.0
        
        # 检查每个草丛的控制状况
        for bush_name, bush_data in self.bush_definitions.items():
            bush_center = bush_data['center']
            
            my_dist = self._calculate_distance_by_pos(my_pos, bush_center)
            enemy_dist = self._calculate_distance_by_pos(enemy_pos, bush_center)
            
            # 距离草丛越近，控制力越强
            if my_dist < enemy_dist:
                control_diff = (enemy_dist - my_dist) / 2000.0
                advantage += min(control_diff, 0.5)
            else:
                control_diff = (my_dist - enemy_dist) / 2000.0
                advantage -= min(control_diff, 0.5)
        
        return max(-1.0, min(advantage, 1.0))
    
    def _calculate_tactical_timing(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算草丛战术时机成熟度"""
        timing = 0.0
        
        # 血量状态：我方血量优势时更适合主动
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        hp_advantage = my_hp_ratio - enemy_hp_ratio
        
        if hp_advantage > 0:
            timing += min(hp_advantage * 2.0, 0.4)
        
        # 技能状态：关键技能可用时时机更好
        if self._has_key_skills_available(my_hero):
            timing += 0.3
        
        # 等级优势：等级领先时更适合主动
        my_level = float(my_hero.get("actor_state", {}).get("level", 1))
        enemy_level = float(enemy_hero.get("actor_state", {}).get("level", 1))
        level_advantage = (my_level - enemy_level) / 3.0  # 归一化
        
        timing += min(level_advantage, 0.3)
        
        return max(0.0, min(timing, 1.0))
    
    def _calculate_intimidation_strength(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算威慑效果强度"""
        strength = 0.0
        
        my_pos = self._get_position(my_hero)
        
        # 在草丛中才有威慑效果
        if self._is_in_any_bush(my_pos):
            strength += 0.4
            
            # 装备和等级优势增加威慑力
            my_level = float(my_hero.get("actor_state", {}).get("level", 1))
            enemy_level = float(enemy_hero.get("actor_state", {}).get("level", 1))
            
            if my_level > enemy_level:
                strength += 0.3
            
            # 血量优势增加威慑力
            my_hp_ratio = self._get_hp_ratio(my_hero)
            enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
            
            if my_hp_ratio > enemy_hp_ratio:
                strength += 0.3
        
        return min(strength, 1.0)
    
    def _calculate_bush_combo_opportunity(self, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算草丛连招机会 (孙尚香特色)"""
        opportunity = 0.0
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 基础条件：我方在草丛边缘或即将进入草丛
        dist_to_bush = self._distance_to_nearest_bush(my_pos)
        if dist_to_bush <= 300:  # 接近草丛
            opportunity += 0.3
            
            # 敌方在合适的距离内
            distance = self._calculate_distance_by_pos(my_pos, enemy_pos)
            if 400 <= distance <= 800:  # 1技能 + 强化普攻的最佳距离
                opportunity += 0.4
                
                # 1技能可用
                s1_cd = float(my_hero.get("skill_state", {}).get("s1_cd", 0))
                if s1_cd <= 0:
                    opportunity += 0.3
        
        return min(opportunity, 1.0)
    
    def _calculate_aggro_reset_value(self, my_hero: Dict, minion_states: Dict) -> float:
        """计算仇恨重置价值"""
        my_pos = self._get_position(my_hero)
        value = 0.0
        
        # 计算当前受到多少敌方小兵的攻击
        attacking_minions = 0
        enemy_minions = minion_states.get('enemy_minions', [])
        
        for minion in enemy_minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance = self._calculate_distance_by_pos(my_pos, minion_position)
                
                # 在小兵攻击范围内
                if distance <= 600:
                    attacking_minions += 1
        
        # 受到的小兵仇恨越多，重置价值越高
        if attacking_minions >= 2:
            value = min(attacking_minions / 5.0, 1.0)
        
        return value
    
    def _calculate_bush_game_initiative(self, my_hero: Dict, enemy_hero: Dict, frame_no: int) -> float:
        """计算草丛博弈主动权"""
        initiative = 0.0
        
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        my_in_bush = self._is_in_any_bush(my_pos)
        enemy_in_bush = self._is_in_any_bush(enemy_pos)
        
        # 我在草丛敌方不在 = 主动权在我
        if my_in_bush and not enemy_in_bush:
            initiative += 0.6
        # 我不在草丛敌方在 = 主动权在敌方
        elif not my_in_bush and enemy_in_bush:
            initiative -= 0.6
        # 双方都在草丛 = 比较其他因素
        elif my_in_bush and enemy_in_bush:
            # 比较血量优势
            my_hp_ratio = self._get_hp_ratio(my_hero)
            enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
            initiative += (my_hp_ratio - enemy_hp_ratio) * 0.5
        
        # 最近的行动频率 (更活跃的一方有主动权)
        activity_advantage = self._calculate_recent_activity_advantage(frame_no)
        initiative += activity_advantage * 0.4
        
        return max(-1.0, min(initiative, 1.0))
    
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
    
    def _calculate_distance_by_pos(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点之间的距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _has_key_skills_available(self, hero: Dict) -> bool:
        """检查关键技能是否可用"""
        skill_state = hero.get("skill_state", {})
        s1_cd = float(skill_state.get("s1_cd", 0))
        s2_cd = float(skill_state.get("s2_cd", 0))
        
        # 至少有一个关键技能可用
        return s1_cd <= 0 or s2_cd <= 0
    
    def _extract_minion_states(self, npcs: List) -> Dict:
        """提取小兵状态"""
        minion_states = {
            'my_minions': [],
            'enemy_minions': []
        }
        
        for npc in npcs:
            if 'SOLDIER' in npc.get('sub_type', ''):
                if npc.get('camp') == self.main_camp:
                    minion_states['my_minions'].append(npc)
                else:
                    minion_states['enemy_minions'].append(npc)
        
        return minion_states
    
    def _calculate_frontline_position(self, my_minions: List[Dict], enemy_minions: List[Dict]) -> float:
        """计算兵线交锋位置"""
        if not my_minions and not enemy_minions:
            return 0.0
        
        all_positions = []
        for minion in my_minions + enemy_minions:
            pos = minion.get('location', {})
            if pos:
                all_positions.append(float(pos.get('x', 0)))
        
        if not all_positions:
            return 0.0
        
        frontline_x = np.mean(all_positions)
        normalized_position = (frontline_x - 7500.0) / 7500.0
        return max(-1.0, min(normalized_position, 1.0))
    
    def _calculate_recent_activity_advantage(self, frame_no: int) -> float:
        """计算最近活动优势 (简化实现)"""
        # 简化：基于草丛状态持续时间
        return 0.0
    
    def _update_bush_history(self, my_hero: Dict, enemy_hero: Dict, minion_states: Dict, frame_no: int):
        """更新草丛历史状态"""
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        
        # 更新敌方最后可见位置
        if self._is_visible_to_self(enemy_hero, my_hero, minion_states):
            self.bush_state['last_enemy_seen_position'] = enemy_pos
            self.bush_state['enemy_last_seen_frame'] = frame_no
        
        # 更新在草丛中的持续时间
        if self._is_in_any_bush(my_pos):
            self.bush_state['in_bush_duration'] += 1
        else:
            self.bush_state['in_bush_duration'] = 0
        
        # 记录历史状态
        history_entry = {
            'frame_no': frame_no,
            'my_pos': my_pos,
            'enemy_pos': enemy_pos,
            'my_in_bush': self._is_in_any_bush(my_pos),
            'enemy_in_bush': self._is_in_any_bush(enemy_pos),
            'vision_state': {
                'self_visible_to_enemy': self._is_visible_to_enemy(my_hero, enemy_hero, minion_states),
                'enemy_visible_to_self': self._is_visible_to_self(enemy_hero, my_hero, minion_states)
            }
        }
        
        self.vision_history.append(history_entry)
        self.bush_activity_history.append(history_entry)
        self.enemy_position_history.append({'frame_no': frame_no, 'position': enemy_pos})
