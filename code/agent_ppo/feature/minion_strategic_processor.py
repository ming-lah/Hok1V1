#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Enhanced by Deep RL Analysis

兵线战略处理器
基于深度强化学习分析，实现完整的兵线特征工程

核心理念：
1. 兵线是经济和经验的主要来源
2. 从"反应式"到"规划式"的战略思维
3. 控线、慢推、快推重置的高级策略
4. 时间动态和意图预测的特征设计
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict


class MinionStrategicProcessor:
    """兵线战略处理器 - 实现基于兵线的完整特征工程"""
    
    def __init__(self, camp: str):
        self.main_camp = camp
        self.enemy_camp = "PLAYERCAMP_2" if camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 历史状态追踪 (时间序列特征)
        self.minion_history = deque(maxlen=60)  # 30fps * 2秒 = 60帧历史
        self.position_history = deque(maxlen=150)  # 5秒位置历史
        self.advantage_history = deque(maxlen=300)  # 10秒优势历史
        
        # 兵线战略状态追踪
        self.strategic_state = {
            'current_strategy': 'unknown',  # freezing, slow_pushing, fast_pushing, neutral
            'freeze_duration': 0,           # 控线持续时间
            'super_wave_building': False,   # 是否在积攒超级兵线
            'last_wave_crash_frame': -1,    # 上次兵进塔的帧数
        }
        
        # 游戏常数
        self.game_constants = {
            'minion_gold_values': {
                'melee': 20,     # 近战兵金币
                'ranged': 15,    # 远程兵金币
                'cannon': 60,    # 炮车金币
            },
            'minion_exp_values': {
                'melee': 30,     # 近战兵经验
                'ranged': 25,    # 远程兵经验
                'cannon': 90,    # 炮车经验
            },
            'exp_range': 1200,   # 经验获取范围
            'last_hit_hp_threshold': 0.15,  # 可补刀血量阈值
            'wave_spawn_interval': 1800,     # 兵线刷新间隔 (30fps * 60s)
            'super_wave_threshold': 8,       # 超级兵线阈值
        }
        
        # 区域划分 (基于地图坐标)
        self.zone_definitions = {
            'my_tower_danger': 0,      # 我方塔下危险区
            'my_tower_safe': 1,        # 我方塔前安全区
            'mid_line': 2,             # 中线区
            'enemy_tower_pressure': 3, # 敌方塔前压制区
            'enemy_tower_danger': 4,   # 敌方塔下危险区
        }
        
    def extract_minion_strategic_features(self, observation: Dict, frame_no: int) -> List[float]:
        """提取兵线战略特征 (40维)"""
        if not isinstance(observation, dict):
            return [0.0] * 40
        
        frame_state = observation.get("frame_state", {})
        if not isinstance(frame_state, dict):
            return [0.0] * 40
        
        # 获取英雄和兵线状态
        hero_states = frame_state.get("hero_states", [])
        npc_states = frame_state.get("npc_states", [])
        
        my_hero = self._find_hero_by_camp(hero_states, self.main_camp)
        enemy_hero = self._find_hero_by_camp(hero_states, self.enemy_camp)
        minion_states = self._extract_minion_states(npc_states)
        
        if not my_hero or not enemy_hero:
            return [0.0] * 40
        
        all_features = []
        
        # 1. 聚合特征 (宏观视角) (12维)
        aggregated_features = self._extract_aggregated_features(minion_states, my_hero, enemy_hero)
        all_features.extend(aggregated_features)
        
        # 2. 相对与预测性特征 (战略视角) (8维)
        predictive_features = self._extract_predictive_features(minion_states, my_hero, enemy_hero, frame_no)
        all_features.extend(predictive_features)
        
        # 3. 时间序列特征 (时间动态) (8维)
        temporal_features = self._extract_temporal_features(minion_states, frame_no)
        all_features.extend(temporal_features)
        
        # 4. 兵线构成特征 (战术微操) (6维)
        compositional_features = self._extract_compositional_features(minion_states)
        all_features.extend(compositional_features)
        
        # 5. 区域控制与压制特征 (高级战略) (6维)
        zonal_features = self._extract_zonal_control_features(minion_states, my_hero, enemy_hero)
        all_features.extend(zonal_features)
        
        # 更新历史状态
        self._update_minion_history(minion_states, my_hero, enemy_hero, frame_no)
        
        return all_features[:40]  # 确保返回40维特征
    
    def _extract_aggregated_features(self, minion_states: Dict, my_hero: Dict, enemy_hero: Dict) -> List[float]:
        """提取聚合特征 (宏观视角) (12维)"""
        features = []
        
        my_minions = minion_states.get('my_minions', [])
        enemy_minions = minion_states.get('enemy_minions', [])
        
        # 兵线数量与构成 (6维)
        my_melee_count = sum(1 for m in my_minions if self._is_melee_minion(m))
        my_ranged_count = sum(1 for m in my_minions if self._is_ranged_minion(m))
        my_cannon_count = sum(1 for m in my_minions if self._is_cannon_minion(m))
        
        enemy_melee_count = sum(1 for m in enemy_minions if self._is_melee_minion(m))
        enemy_ranged_count = sum(1 for m in enemy_minions if self._is_ranged_minion(m))
        enemy_cannon_count = sum(1 for m in enemy_minions if self._is_cannon_minion(m))
        
        features.extend([
            min(my_melee_count / 5.0, 1.0),     # 归一化到[0,1]
            min(my_ranged_count / 5.0, 1.0),
            min(my_cannon_count / 2.0, 1.0),
            min(enemy_melee_count / 5.0, 1.0),
            min(enemy_ranged_count / 5.0, 1.0),
            min(enemy_cannon_count / 2.0, 1.0),
        ])
        
        # 兵线总血量 (3维)
        my_wave_hp = sum(float(m.get('hp', 0)) for m in my_minions)
        enemy_wave_hp = sum(float(m.get('hp', 0)) for m in enemy_minions)
        wave_hp_advantage = (my_wave_hp - enemy_wave_hp) / 5000.0  # 归一化
        
        features.extend([
            min(my_wave_hp / 3000.0, 1.0),      # 我方兵线总血量
            min(enemy_wave_hp / 3000.0, 1.0),   # 敌方兵线总血量
            max(-1.0, min(wave_hp_advantage, 1.0))  # 兵线血量优势
        ])
        
        # 兵线交锋位置 (1维)
        frontline_position = self._calculate_frontline_position(my_minions, enemy_minions)
        features.append(frontline_position)
        
        # 潜在经济价值 (2维)
        enemy_total_gold = self._calculate_total_gold_value(enemy_minions)
        enemy_last_hitable_gold = self._calculate_last_hitable_gold(enemy_minions, my_hero)
        
        features.extend([
            min(enemy_total_gold / 300.0, 1.0),        # 敌方兵线总金币价值
            min(enemy_last_hitable_gold / 150.0, 1.0)  # 可补刀金币价值
        ])
        
        return features
    
    def _extract_predictive_features(self, minion_states: Dict, my_hero: Dict, 
                                   enemy_hero: Dict, frame_no: int) -> List[float]:
        """提取相对与预测性特征 (战略视角) (8维)"""
        features = []
        
        my_minions = minion_states.get('my_minions', [])
        enemy_minions = minion_states.get('enemy_minions', [])
        
        # 兵线优势度 (增强版) (3维)
        hp_advantage = self._calculate_hp_advantage(my_minions, enemy_minions)
        position_advantage = self._calculate_position_advantage(my_minions, enemy_minions)
        siege_advantage = self._calculate_siege_advantage(my_minions, enemy_minions)
        
        # 综合兵线优势度
        overall_advantage = 0.4 * hp_advantage + 0.4 * position_advantage + 0.2 * siege_advantage
        
        features.extend([
            hp_advantage,
            position_advantage, 
            max(-1.0, min(overall_advantage, 1.0))  # 综合兵线优势度
        ])
        
        # 兵线到达时间 (2维)
        my_time_to_enemy_tower = self._estimate_time_to_tower(my_minions, 'enemy')
        enemy_time_to_my_tower = self._estimate_time_to_tower(enemy_minions, 'my')
        
        features.extend([
            min(my_time_to_enemy_tower / 30.0, 1.0),    # 归一化到30秒
            min(enemy_time_to_my_tower / 30.0, 1.0)
        ])
        
        # 下一波兵线刷新倒计时 (1维)
        next_wave_timer = self._calculate_next_wave_timer(frame_no)
        features.append(next_wave_timer)
        
        # 精细化特征 (微操视角) (2维)
        nearest_enemy_minion_info = self._get_nearest_enemy_minion_info(enemy_minions, my_hero)
        features.extend(nearest_enemy_minion_info)
        
        return features
    
    def _extract_temporal_features(self, minion_states: Dict, frame_no: int) -> List[float]:
        """提取时间序列特征 (时间动态) (8维)"""
        features = []
        
        if len(self.position_history) < 5:
            return [0.0] * 8
        
        # 当前兵线交锋位置
        current_position = self._calculate_frontline_position(
            minion_states.get('my_minions', []), 
            minion_states.get('enemy_minions', [])
        )
        
        # 兵线位置历史/趋势 (4维)
        recent_positions = [entry['frontline_position'] for entry in list(self.position_history)[-30:]]  # 1秒历史
        older_positions = [entry['frontline_position'] for entry in list(self.position_history)[-150:-120]]  # 4-5秒前
        
        if recent_positions and older_positions:
            recent_avg = np.mean(recent_positions)
            older_avg = np.mean(older_positions)
            
            # 位置趋势
            position_trend = (recent_avg - older_avg) / 2.0  # 归一化
            position_volatility = np.std(recent_positions) if len(recent_positions) > 1 else 0.0
            
            features.extend([
                recent_avg,              # 近期平均位置
                older_avg,               # 历史平均位置
                position_trend,          # 位置变化趋势
                min(position_volatility, 1.0)  # 位置波动性
            ])
        else:
            features.extend([current_position, current_position, 0.0, 0.0])
        
        # 兵线优势度变化率 (2维)
        if len(self.advantage_history) >= 150:  # 5秒历史
            current_advantage = self.advantage_history[-1]
            past_advantage = self.advantage_history[-150]
            advantage_change_rate = (current_advantage - past_advantage) / 5.0  # 每秒变化率
            
            features.extend([
                current_advantage,                           # 当前优势度
                max(-1.0, min(advantage_change_rate, 1.0))  # 优势变化率
            ])
        else:
            features.extend([0.0, 0.0])
        
        # 战略状态持续时间 (2维)
        freeze_duration_normalized = min(self.strategic_state['freeze_duration'] / 300.0, 1.0)  # 10秒归一化
        time_since_wave_crash = min((frame_no - self.strategic_state['last_wave_crash_frame']) / 600.0, 1.0)  # 20秒归一化
        
        features.extend([
            freeze_duration_normalized,
            time_since_wave_crash
        ])
        
        return features
    
    def _extract_compositional_features(self, minion_states: Dict) -> List[float]:
        """提取兵线构成特征 (战术微操) (6维)"""
        features = []
        
        my_minions = minion_states.get('my_minions', [])
        enemy_minions = minion_states.get('enemy_minions', [])
        
        # 我方兵线构成比例 (3维)
        my_total = len(my_minions)
        if my_total > 0:
            my_melee_ratio = sum(1 for m in my_minions if self._is_melee_minion(m)) / my_total
            my_ranged_ratio = sum(1 for m in my_minions if self._is_ranged_minion(m)) / my_total
            my_cannon_ratio = sum(1 for m in my_minions if self._is_cannon_minion(m)) / my_total
        else:
            my_melee_ratio = my_ranged_ratio = my_cannon_ratio = 0.0
        
        features.extend([my_melee_ratio, my_ranged_ratio, my_cannon_ratio])
        
        # 敌方兵线构成比例 (3维)
        enemy_total = len(enemy_minions)
        if enemy_total > 0:
            enemy_melee_ratio = sum(1 for m in enemy_minions if self._is_melee_minion(m)) / enemy_total
            enemy_ranged_ratio = sum(1 for m in enemy_minions if self._is_ranged_minion(m)) / enemy_total
            enemy_cannon_ratio = sum(1 for m in enemy_minions if self._is_cannon_minion(m)) / enemy_total
        else:
            enemy_melee_ratio = enemy_ranged_ratio = enemy_cannon_ratio = 0.0
        
        features.extend([enemy_melee_ratio, enemy_ranged_ratio, enemy_cannon_ratio])
        
        return features
    
    def _extract_zonal_control_features(self, minion_states: Dict, my_hero: Dict, enemy_hero: Dict) -> List[float]:
        """提取区域控制与压制特征 (高级战略) (6维)"""
        features = []
        
        my_minions = minion_states.get('my_minions', [])
        enemy_minions = minion_states.get('enemy_minions', [])
        
        # 兵线交汇点区域划分 (5维独热编码)
        frontline_position = self._calculate_frontline_position(my_minions, enemy_minions)
        current_zone = self._determine_zone_from_position(frontline_position)
        
        zone_encoding = [0.0] * 5
        if 0 <= current_zone < 5:
            zone_encoding[current_zone] = 1.0
        
        features.extend(zone_encoding)
        
        # 敌方英雄与经验区的关系 (1维)
        enemy_in_exp_deny_position = self._is_enemy_in_exp_deny_position(enemy_hero, enemy_minions)
        features.append(1.0 if enemy_in_exp_deny_position else 0.0)
        
        return features
    
    # ============ 核心计算方法 ============
    
    def _calculate_frontline_position(self, my_minions: List[Dict], enemy_minions: List[Dict]) -> float:
        """计算兵线交锋位置 (-1: 我方塔下, 0: 中线, +1: 敌方塔下)"""
        if not my_minions and not enemy_minions:
            return 0.0
        
        all_positions = []
        
        # 收集所有小兵的x坐标 (假设兵线沿x轴移动)
        for minion in my_minions + enemy_minions:
            pos = minion.get('location', {})
            if pos:
                all_positions.append(float(pos.get('x', 0)))
        
        if not all_positions:
            return 0.0
        
        # 计算重心位置
        frontline_x = np.mean(all_positions)
        
        # 映射到[-1, 1]范围 (假设地图x范围为[0, 15000])
        normalized_position = (frontline_x - 7500.0) / 7500.0
        return max(-1.0, min(normalized_position, 1.0))
    
    def _calculate_hp_advantage(self, my_minions: List[Dict], enemy_minions: List[Dict]) -> float:
        """计算血量优势"""
        my_hp = sum(float(m.get('hp', 0)) for m in my_minions)
        enemy_hp = sum(float(m.get('hp', 0)) for m in enemy_minions)
        
        if my_hp + enemy_hp == 0:
            return 0.0
        
        return (my_hp - enemy_hp) / (my_hp + enemy_hp)
    
    def _calculate_position_advantage(self, my_minions: List[Dict], enemy_minions: List[Dict]) -> float:
        """计算位置优势"""
        frontline_pos = self._calculate_frontline_position(my_minions, enemy_minions)
        # 正值表示兵线靠近敌方(我方优势)，负值表示兵线靠近我方(敌方优势)
        return frontline_pos
    
    def _calculate_siege_advantage(self, my_minions: List[Dict], enemy_minions: List[Dict]) -> float:
        """计算攻城优势 (炮车等关键单位)"""
        my_cannons = sum(1 for m in my_minions if self._is_cannon_minion(m))
        enemy_cannons = sum(1 for m in enemy_minions if self._is_cannon_minion(m))
        
        return (my_cannons - enemy_cannons) / 3.0  # 归一化
    
    def _estimate_time_to_tower(self, minions: List[Dict], tower_type: str) -> float:
        """估算兵线到达防御塔的时间"""
        if not minions:
            return 30.0  # 最大值
        
        # 找到最前面的小兵
        if tower_type == 'enemy':
            # 到敌方塔：找x坐标最大的小兵
            front_minion = max(minions, key=lambda m: float(m.get('location', {}).get('x', 0)), default=None)
            target_x = 13000.0  # 假设敌方塔位置
        else:
            # 到我方塔：找x坐标最小的小兵
            front_minion = min(minions, key=lambda m: float(m.get('location', {}).get('x', 0)), default=None)
            target_x = 2000.0   # 假设我方塔位置
        
        if not front_minion:
            return 30.0
        
        current_x = float(front_minion.get('location', {}).get('x', 0))
        distance = abs(target_x - current_x)
        
        # 假设小兵移动速度为100单位/秒，30fps
        minion_speed = 100.0 / 30.0  # 每帧移动距离
        estimated_frames = distance / max(minion_speed, 1.0)
        estimated_seconds = estimated_frames / 30.0
        
        return min(estimated_seconds, 30.0)
    
    def _calculate_next_wave_timer(self, frame_no: int) -> float:
        """计算下一波兵线刷新倒计时"""
        time_since_spawn = frame_no % self.game_constants['wave_spawn_interval']
        time_to_next = self.game_constants['wave_spawn_interval'] - time_since_spawn
        return time_to_next / self.game_constants['wave_spawn_interval']  # 归一化到[0,1]
    
    def _get_nearest_enemy_minion_info(self, enemy_minions: List[Dict], my_hero: Dict) -> List[float]:
        """获取最近敌方小兵信息 (2维)"""
        if not enemy_minions:
            return [0.0, 0.0]
        
        my_pos = self._get_position(my_hero)
        
        # 找到最近的敌方小兵
        nearest_minion = None
        min_distance = float('inf')
        
        for minion in enemy_minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                pos = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance = self._calculate_distance_by_pos(my_pos, pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_minion = minion
        
        if not nearest_minion:
            return [0.0, 0.0]
        
        # 最近小兵的血量比例
        hp_ratio = float(nearest_minion.get('hp', 0)) / max(float(nearest_minion.get('max_hp', 1)), 1)
        
        # 是否处于可补刀状态
        is_last_hitable = 1.0 if hp_ratio < self.game_constants['last_hit_hp_threshold'] else 0.0
        
        return [hp_ratio, is_last_hitable]
    
    def _calculate_total_gold_value(self, minions: List[Dict]) -> float:
        """计算兵线总金币价值"""
        total_gold = 0.0
        
        for minion in minions:
            if self._is_melee_minion(minion):
                total_gold += self.game_constants['minion_gold_values']['melee']
            elif self._is_ranged_minion(minion):
                total_gold += self.game_constants['minion_gold_values']['ranged']
            elif self._is_cannon_minion(minion):
                total_gold += self.game_constants['minion_gold_values']['cannon']
        
        return total_gold
    
    def _calculate_last_hitable_gold(self, enemy_minions: List[Dict], my_hero: Dict) -> float:
        """计算可补刀的金币价值"""
        last_hitable_gold = 0.0
        my_pos = self._get_position(my_hero)
        
        for minion in enemy_minions:
            hp_ratio = float(minion.get('hp', 0)) / max(float(minion.get('max_hp', 1)), 1)
            
            if hp_ratio < self.game_constants['last_hit_hp_threshold']:
                minion_pos = minion.get('location', {})
                if minion_pos:
                    pos = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                    distance = self._calculate_distance_by_pos(my_pos, pos)
                    
                    if distance <= 1200:  # 在攻击范围内
                        if self._is_melee_minion(minion):
                            last_hitable_gold += self.game_constants['minion_gold_values']['melee']
                        elif self._is_ranged_minion(minion):
                            last_hitable_gold += self.game_constants['minion_gold_values']['ranged']
                        elif self._is_cannon_minion(minion):
                            last_hitable_gold += self.game_constants['minion_gold_values']['cannon']
        
        return last_hitable_gold
    
    def _determine_zone_from_position(self, frontline_position: float) -> int:
        """根据前线位置确定区域"""
        if frontline_position < -0.6:
            return self.zone_definitions['my_tower_danger']
        elif frontline_position < -0.2:
            return self.zone_definitions['my_tower_safe']
        elif frontline_position < 0.2:
            return self.zone_definitions['mid_line']
        elif frontline_position < 0.6:
            return self.zone_definitions['enemy_tower_pressure']
        else:
            return self.zone_definitions['enemy_tower_danger']
    
    def _is_enemy_in_exp_deny_position(self, enemy_hero: Dict, enemy_minions: List[Dict]) -> bool:
        """判断敌方英雄是否处于被经验压制的位置"""
        if not enemy_minions:
            return False
        
        enemy_pos = self._get_position(enemy_hero)
        
        # 检查是否有敌方小兵在经验范围外
        for minion in enemy_minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                pos = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance = self._calculate_distance_by_pos(enemy_pos, pos)
                
                if distance > self.game_constants['exp_range']:
                    return True
        
        return False
    
    # ============ 兵种识别方法 ============
    
    def _is_melee_minion(self, minion: Dict) -> bool:
        """判断是否为近战小兵"""
        sub_type = minion.get('sub_type', '')
        return 'SOLDIER_LINE' in sub_type and 'MELEE' in sub_type
    
    def _is_ranged_minion(self, minion: Dict) -> bool:
        """判断是否为远程小兵"""
        sub_type = minion.get('sub_type', '')
        return 'SOLDIER_LINE' in sub_type and 'REMOTE' in sub_type
    
    def _is_cannon_minion(self, minion: Dict) -> bool:
        """判断是否为炮车"""
        sub_type = minion.get('sub_type', '')
        return 'SOLDIER_CANNON' in sub_type
    
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
    
    def _calculate_distance_by_pos(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点之间的距离"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
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
    
    def _update_minion_history(self, minion_states: Dict, my_hero: Dict, enemy_hero: Dict, frame_no: int):
        """更新兵线历史状态"""
        my_minions = minion_states.get('my_minions', [])
        enemy_minions = minion_states.get('enemy_minions', [])
        
        # 计算当前状态
        frontline_position = self._calculate_frontline_position(my_minions, enemy_minions)
        overall_advantage = self._calculate_hp_advantage(my_minions, enemy_minions)
        
        # 更新位置历史
        position_entry = {
            'frame_no': frame_no,
            'frontline_position': frontline_position
        }
        self.position_history.append(position_entry)
        
        # 更新优势历史
        self.advantage_history.append(overall_advantage)
        
        # 更新战略状态
        self._update_strategic_state(minion_states, frontline_position, frame_no)
        
        # 更新整体历史
        history_entry = {
            'frame_no': frame_no,
            'minion_states': minion_states,
            'frontline_position': frontline_position,
            'overall_advantage': overall_advantage,
            'strategic_state': self.strategic_state.copy()
        }
        self.minion_history.append(history_entry)
    
    def _update_strategic_state(self, minion_states: Dict, frontline_position: float, frame_no: int):
        """更新战略状态"""
        my_minions = minion_states.get('my_minions', [])
        enemy_minions = minion_states.get('enemy_minions', [])
        
        # 检测控线状态
        if -0.4 < frontline_position < -0.1:  # 在我方塔前安全区
            self.strategic_state['freeze_duration'] += 1
            if self.strategic_state['freeze_duration'] > 30:  # 持续1秒以上
                self.strategic_state['current_strategy'] = 'freezing'
        else:
            self.strategic_state['freeze_duration'] = 0
        
        # 检测超级兵线
        if len(my_minions) >= self.game_constants['super_wave_threshold']:
            self.strategic_state['super_wave_building'] = True
            self.strategic_state['current_strategy'] = 'slow_pushing'
        else:
            self.strategic_state['super_wave_building'] = False
        
        # 检测兵线撞塔
        if frontline_position > 0.6:  # 兵线到达敌方塔下
            self.strategic_state['last_wave_crash_frame'] = frame_no
            if self.strategic_state['current_strategy'] != 'slow_pushing':
                self.strategic_state['current_strategy'] = 'fast_pushing'
        
        # 默认中性状态
        if self.strategic_state['freeze_duration'] == 0 and not self.strategic_state['super_wave_building']:
            self.strategic_state['current_strategy'] = 'neutral'
