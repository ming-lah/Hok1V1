#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Enhanced by Deep RL Analysis

防御塔战略奖励系统
基于深度强化学习分析，实现完整的防御塔奖励函数

核心理念：
1. 终端奖励：胜负的明确信号
2. 事件驱动奖励：关键行为的即时反馈
3. 势能奖励：平滑的状态改善奖励
4. 动态权重：基于兵线和局势的智能调整
5. 机会成本：引导最优的"发育"vs"推塔"决策
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict


class TowerStrategicRewards:
    """防御塔战略奖励系统 - 实现完整的推塔奖励函数"""
    
    def __init__(self, main_hero_id: int, main_camp: str):
        self.main_hero_id = main_hero_id
        self.main_camp = main_camp
        self.enemy_camp = "PLAYERCAMP_2" if main_camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 历史状态追踪
        self.state_history = deque(maxlen=10)
        
        # 奖励权重配置 - 针对拆塔胜利条件优化
        self.reward_weights = {
            # 1. 终端奖励 (Terminal Rewards) - 大幅提升拆塔胜利的奖励
            'tower_victory': 2000.0,            # 摧毁敌方防御塔 (胜利) - 提升到2000
            'tower_defeat': -2000.0,            # 我方防御塔被摧毁 (失败) - 提升到-2000
            
            # 2. 事件驱动型奖励 (Event-Driven Rewards)
            'enemy_tower_damage': 8.0,          # 对敌方防御塔造成伤害 (每点伤害) - 从5.0提升到8.0
            'my_tower_damage': -8.0,            # 我方防御塔受到伤害 - 从-5.0提升到-8.0
            'minion_tower_damage': 3.0,         # 我方小兵对敌方塔造成伤害 - 从2.0提升到3.0
            'tower_shot_penalty': -25.0,        # 被敌方防御塔攻击的严重惩罚 - 从-20.0提升到-25.0
            
            # 3. 基于势能的奖励 (Potential-Based Rewards)
            'tower_hp_potential': 100.0,        # 防御塔血量差势能权重
            
            # 4. 兵线运营奖励 (Wave Management Rewards)
            'wave_crash_bonus': 50.0,           # 送兵进塔奖励
            'perfect_clear_bonus': 30.0,        # 完美解线奖励
            'wave_control_bonus': 20.0,         # 兵线控制奖励
            
            # 5. 智能推塔奖励 (Smart Pushing Rewards)
            'smart_push_multiplier': 2.0,       # 有兵线优势时推塔的倍数加成
            'blind_push_penalty': 0.5,          # 无兵线强推的惩罚倍数
            
            # 6. 动态发育奖励 (Dynamic Farming Rewards)
            'pressure_farming_penalty': -10.0,  # 防守压力下贪图发育的惩罚
            'safe_farming_bonus': 5.0,         # 安全发育奖励
            
            # 7. 【新增】胜利条件导向奖励 (Victory-Focused Rewards)
            'endgame_urgency_bonus': 50.0,      # 终局紧迫状态下的行动奖励
            'one_push_victory_bonus': 100.0,    # 一波结束游戏的奖励
            'desperate_defense_bonus': 80.0,    # 绝境防守成功的奖励
            'tower_hp_amplified_bonus': 200.0,  # 终局阶段塔血量优势的放大奖励
            'decisive_timing_bonus': 60.0,      # 把握决战时机的奖励
            'critical_tower_save': 150.0,       # 拯救危险防御塔的奖励
            'finishing_blow_bonus': 300.0,      # 最后一击摧毁敌方塔的额外奖励
            
            # 8. 战略失误惩罚 (Strategic Blunder Penalties)
            'greedy_death_penalty': -500.0,     # 贪婪致死的严重惩罚
            'missed_opportunity_penalty': -100.0, # 错失推塔机会的惩罚
            'poor_timing_penalty': -50.0        # 时机选择不当的惩罚
        }
        
        # 势能函数历史
        self.potential_history = deque(maxlen=2)
        
        # 兵线状态追踪
        self.wave_tracker = WaveStateTracker()
        
        # 战略决策分析器
        self.strategic_analyzer = StrategicDecisionAnalyzer()
        
    def calculate_tower_strategic_rewards(self, frame_data: Dict, main_hero: Dict, 
                                        enemy_hero: Dict, frame_no: int, 
                                        tower_features: List[float] = None) -> Dict[str, float]:
        """计算防御塔战略奖励"""
        if not main_hero or not enemy_hero:
            return {}
        
        rewards = {}
        
        # 提取当前状态
        current_state = self._extract_current_state(frame_data, main_hero, enemy_hero)
        
        # 1. 终端奖励
        terminal_rewards = self._calculate_terminal_rewards(frame_data, main_hero, enemy_hero)
        rewards.update(terminal_rewards)
        
        # 2. 事件驱动奖励
        event_rewards = self._calculate_event_driven_rewards(frame_data, main_hero, enemy_hero, current_state)
        rewards.update(event_rewards)
        
        # 3. 势能奖励
        potential_rewards = self._calculate_potential_rewards(current_state)
        rewards.update(potential_rewards)
        
        # 4. 兵线运营奖励
        wave_rewards = self._calculate_wave_management_rewards(frame_data, current_state)
        rewards.update(wave_rewards)
        
        # 5. 智能推塔奖励 (基于您的动态权重思路)
        smart_push_rewards = self._calculate_smart_pushing_rewards(frame_data, main_hero, current_state)
        rewards.update(smart_push_rewards)
        
        # 6. 动态发育奖励 (基于您的机会成本思路)
        dynamic_farming_rewards = self._calculate_dynamic_farming_rewards(frame_data, main_hero, current_state)
        rewards.update(dynamic_farming_rewards)
        
        # 7. 战略失误惩罚
        strategic_penalties = self._calculate_strategic_penalties(frame_data, main_hero, enemy_hero, current_state)
        rewards.update(strategic_penalties)
        
        # 8. 【新增】胜利条件导向奖励
        if tower_features and len(tower_features) >= 35:
            victory_focused_rewards = self._calculate_victory_focused_rewards(
                tower_features[-5:], current_state, main_hero, enemy_hero, frame_data
            )
            rewards.update(victory_focused_rewards)
        
        # 更新历史状态
        self._update_state_history(current_state, frame_no)
        
        return rewards
    
    def _calculate_terminal_rewards(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict[str, float]:
        """计算终端奖励"""
        rewards = {}
        
        # 检查防御塔的摧毁
        for action in frame_data.get("frame_action", []):
            dead_action = action.get("dead_action", {})
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            
            if "TOWER" in str(death.get("type", "")):
                if death.get("camp") != self.main_camp:  # 摧毁敌方塔
                    rewards['tower_victory'] = self.reward_weights['tower_victory']
                else:  # 我方塔被摧毁
                    rewards['tower_defeat'] = self.reward_weights['tower_defeat']
        
        return rewards
    
    def _calculate_event_driven_rewards(self, frame_data: Dict, main_hero: Dict, 
                                      enemy_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算事件驱动奖励"""
        rewards = {}
        
        if not self.state_history:
            return rewards
        
        prev_state = self.state_history[-1]
        
        # 计算防御塔血量变化
        my_tower_hp_change = prev_state['my_tower_hp'] - current_state['my_tower_hp']
        enemy_tower_hp_change = prev_state['enemy_tower_hp'] - current_state['enemy_tower_hp']
        
        # 对敌方防御塔造成伤害
        if enemy_tower_hp_change > 0:
            # 检查是否是英雄造成的伤害
            hero_damage = self._check_hero_tower_damage(frame_data, main_hero, 'enemy')
            if hero_damage > 0:
                rewards['enemy_tower_damage'] = hero_damage * self.reward_weights['enemy_tower_damage']
            
            # 检查是否是小兵造成的伤害
            minion_damage = enemy_tower_hp_change - hero_damage
            if minion_damage > 0:
                rewards['minion_tower_damage'] = minion_damage * self.reward_weights['minion_tower_damage']
        
        # 我方防御塔受到伤害
        if my_tower_hp_change > 0:
            rewards['my_tower_damage'] = my_tower_hp_change * self.reward_weights['my_tower_damage']
        
        # 被敌方防御塔攻击的惩罚
        tower_shot_damage = self._check_tower_shot_damage(frame_data, main_hero)
        if tower_shot_damage > 0:
            rewards['tower_shot_penalty'] = tower_shot_damage * self.reward_weights['tower_shot_penalty']
        
        return rewards
    
    def _calculate_potential_rewards(self, current_state: Dict) -> Dict[str, float]:
        """计算基于势能的奖励 (Φ(s_t) - Φ(s_{t-1}))"""
        rewards = {}
        
        if not self.potential_history:
            # 首次计算，记录当前势能
            current_potential = self._calculate_potential_function(current_state)
            self.potential_history.append(current_potential)
            return rewards
        
        # 计算势能变化
        prev_potential = self.potential_history[-1]
        current_potential = self._calculate_potential_function(current_state)
        
        potential_change = current_potential - prev_potential
        
        if abs(potential_change) > 0.001:  # 避免微小变化的噪音
            rewards['tower_hp_potential'] = potential_change * self.reward_weights['tower_hp_potential']
        
        # 更新势能历史
        self.potential_history.append(current_potential)
        
        return rewards
    
    def _calculate_wave_management_rewards(self, frame_data: Dict, current_state: Dict) -> Dict[str, float]:
        """计算兵线运营奖励"""
        rewards = {}
        
        # 更新兵线追踪器
        self.wave_tracker.update(current_state)
        
        # 送兵进塔奖励
        if self.wave_tracker.detect_wave_crash_to_enemy_tower():
            rewards['wave_crash_bonus'] = self.reward_weights['wave_crash_bonus']
        
        # 完美解线奖励
        perfect_clear_score = self.wave_tracker.evaluate_perfect_clear()
        if perfect_clear_score > 0:
            rewards['perfect_clear_bonus'] = perfect_clear_score * self.reward_weights['perfect_clear_bonus']
        
        # 兵线控制奖励
        wave_control_score = self.wave_tracker.evaluate_wave_control()
        if wave_control_score > 0:
            rewards['wave_control_bonus'] = wave_control_score * self.reward_weights['wave_control_bonus']
        
        return rewards
    
    def _calculate_smart_pushing_rewards(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算智能推塔奖励 (基于您的动态权重思路)"""
        rewards = {}
        
        # 检查英雄是否对敌方塔造成了伤害
        hero_tower_damage = self._check_hero_tower_damage(frame_data, main_hero, 'enemy')
        
        if hero_tower_damage > 0:
            # 获取兵线优势度
            wave_advantage = current_state.get('wave_advantage', 0.0)
            my_minions_at_enemy_tower = current_state.get('my_minions_at_enemy_tower', 0)
            
            # 基础推塔奖励
            base_reward = hero_tower_damage * self.reward_weights['enemy_tower_damage']
            
            # 根据兵线状态调整奖励倍数
            if wave_advantage > 0.3 or my_minions_at_enemy_tower > 0:
                # 有兵线优势时推塔 - 智能推塔
                multiplier = self.reward_weights['smart_push_multiplier']
                adjusted_reward = base_reward * multiplier
                rewards['smart_tower_push'] = adjusted_reward - base_reward  # 额外奖励
            elif wave_advantage < -0.2:
                # 无兵线强推 - 盲目推塔
                penalty_multiplier = self.reward_weights['blind_push_penalty']
                penalty = base_reward * (1 - penalty_multiplier)
                rewards['blind_push_penalty'] = -penalty
        
        return rewards
    
    def _calculate_dynamic_farming_rewards(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算动态发育奖励 (基于您的机会成本思路)"""
        rewards = {}
        
        # 检查金币获取
        gold_gained = self._check_gold_gain(frame_data, main_hero)
        
        if gold_gained > 0:
            # 获取防守压力
            defensive_pressure = current_state.get('defensive_pressure', 0.0)
            my_tower_hp_ratio = current_state.get('my_tower_hp', 1.0)
            
            # 基础发育奖励
            base_farming_reward = gold_gained * 0.1  # 基础系数
            
            # 根据防守压力调整奖励
            if defensive_pressure > 0.5 and my_tower_hp_ratio < 0.6:
                # 防守压力大时仍然贪图发育 - 惩罚
                pressure_penalty = defensive_pressure * self.reward_weights['pressure_farming_penalty']
                rewards['pressure_farming_penalty'] = pressure_penalty
            elif defensive_pressure < 0.2:
                # 安全发育 - 奖励
                safe_bonus = base_farming_reward * self.reward_weights['safe_farming_bonus'] * 0.1
                rewards['safe_farming_bonus'] = safe_bonus
        
        return rewards
    
    def _calculate_strategic_penalties(self, frame_data: Dict, main_hero: Dict, 
                                     enemy_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算战略失误惩罚"""
        rewards = {}
        
        # 检查英雄死亡
        if self._check_hero_death(frame_data, main_hero):
            # 检查是否是"贪婪致死"
            if self._is_greedy_death(current_state, main_hero):
                rewards['greedy_death_penalty'] = self.reward_weights['greedy_death_penalty']
        
        # 检查错失推塔机会
        missed_opportunity_score = self._evaluate_missed_opportunity(current_state)
        if missed_opportunity_score > 0:
            rewards['missed_opportunity_penalty'] = missed_opportunity_score * self.reward_weights['missed_opportunity_penalty']
        
        # 检查时机选择不当
        poor_timing_score = self._evaluate_poor_timing(frame_data, main_hero, current_state)
        if poor_timing_score > 0:
            rewards['poor_timing_penalty'] = poor_timing_score * self.reward_weights['poor_timing_penalty']
        
        return rewards
    
    def _extract_current_state(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict:
        """提取当前状态"""
        # 获取防御塔状态
        tower_states = self._extract_tower_states(frame_data.get("npc_states", []))
        minion_states = self._extract_minion_states(frame_data.get("npc_states", []))
        
        # 计算兵线相关指标
        my_tower_pos = tower_states.get('my_tower_pos', (0, 0))
        enemy_tower_pos = tower_states.get('enemy_tower_pos', (0, 0))
        
        my_wave_potential = self._calculate_wave_offensive_potential(
            minion_states.get('my_minions', []), enemy_tower_pos
        )
        enemy_wave_pressure = self._calculate_wave_defensive_pressure(
            minion_states.get('enemy_minions', []), my_tower_pos
        )
        
        return {
            'my_tower_hp': tower_states.get('my_tower_hp_ratio', 1.0),
            'enemy_tower_hp': tower_states.get('enemy_tower_hp_ratio', 1.0),
            'my_tower_pos': my_tower_pos,
            'enemy_tower_pos': enemy_tower_pos,
            'my_minions_at_enemy_tower': self._count_minions_at_tower(
                minion_states.get('my_minions', []), enemy_tower_pos
            ),
            'enemy_minions_at_my_tower': self._count_minions_at_tower(
                minion_states.get('enemy_minions', []), my_tower_pos
            ),
            'wave_advantage': my_wave_potential - enemy_wave_pressure,
            'defensive_pressure': enemy_wave_pressure,
            'my_hero_pos': self._get_position(main_hero),
            'enemy_hero_pos': self._get_position(enemy_hero),
            'my_hp_ratio': self._get_hp_ratio(main_hero),
            'enemy_hp_ratio': self._get_hp_ratio(enemy_hero),
        }
    
    def _calculate_potential_function(self, state: Dict) -> float:
        """计算势能函数 Φ(s) = w * (我方塔HP - 敌方塔HP)"""
        my_tower_hp = state.get('my_tower_hp', 1.0)
        enemy_tower_hp = state.get('enemy_tower_hp', 1.0)
        
        # 势能 = 塔血量差
        potential = my_tower_hp - enemy_tower_hp
        return potential
    
    def _check_hero_tower_damage(self, frame_data: Dict, hero: Dict, tower_type: str) -> float:
        """检查英雄对防御塔造成的伤害"""
        # 简化实现：通过hurt_action检查
        hero_id = hero.get("player_id", -1)
        total_damage = 0.0
        
        for action in frame_data.get("frame_action", []):
            hurt_action = action.get("hurt_action", {})
            if hurt_action:
                attacker_id = hurt_action.get("attacker_player_id", -1)
                target_type = hurt_action.get("target_type", "")
                damage = float(hurt_action.get("damage", 0))
                
                if attacker_id == hero_id and "TOWER" in target_type:
                    total_damage += damage
        
        return total_damage
    
    def _check_tower_shot_damage(self, frame_data: Dict, hero: Dict) -> float:
        """检查防御塔对英雄造成的伤害"""
        hero_id = hero.get("player_id", -1)
        total_damage = 0.0
        
        for action in frame_data.get("frame_action", []):
            hurt_action = action.get("hurt_action", {})
            if hurt_action:
                target_player_id = hurt_action.get("target_player_id", -1)
                attacker_type = hurt_action.get("attacker_type", "")
                damage = float(hurt_action.get("damage", 0))
                
                if target_player_id == hero_id and "TOWER" in attacker_type:
                    total_damage += damage
        
        return total_damage
    
    def _check_gold_gain(self, frame_data: Dict, hero: Dict) -> float:
        """检查金币获取"""
        if not self.state_history:
            return 0.0
        
        current_gold = float(hero.get("money", 0))
        prev_gold = self.state_history[-1].get('hero_gold', current_gold)
        
        return max(0.0, current_gold - prev_gold)
    
    def _check_hero_death(self, frame_data: Dict, hero: Dict) -> bool:
        """检查英雄是否死亡"""
        hero_id = hero.get("player_id", -1)
        
        for action in frame_data.get("frame_action", []):
            dead_action = action.get("dead_action", {})
            death = dead_action.get("death", {})
            
            if death.get("player_id") == hero_id:
                return True
        
        return False
    
    def _is_greedy_death(self, current_state: Dict, hero: Dict) -> bool:
        """判断是否是贪婪致死"""
        my_tower_hp = current_state.get('my_tower_hp', 1.0)
        hero_pos = current_state.get('my_hero_pos', (0, 0))
        enemy_tower_pos = current_state.get('enemy_tower_pos', (0, 0))
        
        # 我方塔血量低于安全线
        if my_tower_hp < 0.4:
            # 英雄仍在敌方塔下
            distance_to_enemy_tower = self._calculate_distance_by_pos(hero_pos, enemy_tower_pos)
            if distance_to_enemy_tower < 1200:
                return True
        
        return False
    
    def _evaluate_missed_opportunity(self, current_state: Dict) -> float:
        """评估错失推塔机会"""
        # 检查是否有好的推塔机会但没有利用
        wave_advantage = current_state.get('wave_advantage', 0.0)
        my_minions_at_enemy_tower = current_state.get('my_minions_at_enemy_tower', 0)
        
        my_hero_pos = current_state.get('my_hero_pos', (0, 0))
        enemy_tower_pos = current_state.get('enemy_tower_pos', (0, 0))
        distance_to_enemy_tower = self._calculate_distance_by_pos(my_hero_pos, enemy_tower_pos)
        
        # 有兵线优势且小兵在敌方塔下，但英雄距离很远
        if wave_advantage > 0.5 and my_minions_at_enemy_tower > 0 and distance_to_enemy_tower > 2000:
            return wave_advantage * my_minions_at_enemy_tower / 3.0
        
        return 0.0
    
    def _evaluate_poor_timing(self, frame_data: Dict, hero: Dict, current_state: Dict) -> float:
        """评估时机选择不当"""
        # 检查在不合适的时机做出的行为
        score = 0.0
        
        # 在防守压力大时强行推塔
        defensive_pressure = current_state.get('defensive_pressure', 0.0)
        my_tower_hp = current_state.get('my_tower_hp', 1.0)
        
        hero_tower_damage = self._check_hero_tower_damage(frame_data, hero, 'enemy')
        
        if hero_tower_damage > 0 and defensive_pressure > 0.7 and my_tower_hp < 0.5:
            score += defensive_pressure
        
        return score
    
    # ============ 辅助方法 ============
    
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
    
    def _extract_tower_states(self, npcs: List) -> Dict:
        """提取防御塔状态"""
        tower_states = {
            'my_tower_hp_ratio': 1.0,
            'enemy_tower_hp_ratio': 1.0,
            'my_tower_pos': (0, 0),
            'enemy_tower_pos': (0, 0)
        }
        
        for npc in npcs:
            if 'TOWER' in npc.get('sub_type', ''):
                hp_ratio = float(npc.get('hp', 0)) / max(float(npc.get('max_hp', 1)), 1)
                npc_pos = npc.get('location', {})
                position = (float(npc_pos.get('x', 0)), float(npc_pos.get('z', 0)))
                
                if npc.get('camp') == self.main_camp:
                    tower_states['my_tower_hp_ratio'] = hp_ratio
                    tower_states['my_tower_pos'] = position
                else:
                    tower_states['enemy_tower_hp_ratio'] = hp_ratio
                    tower_states['enemy_tower_pos'] = position
        
        return tower_states
    
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
    
    def _calculate_wave_offensive_potential(self, minions: List[Dict], tower_pos: Tuple[float, float]) -> float:
        """计算兵线进攻潜力"""
        if not minions:
            return 0.0
        
        total_potential = 0.0
        for minion in minions:
            minion_hp = float(minion.get('hp', 0))
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance = self._calculate_distance_by_pos(minion_position, tower_pos)
                
                if distance > 0:
                    potential = minion_hp / max(distance, 100.0)
                    total_potential += potential
        
        return min(total_potential / 1000.0, 1.0)
    
    def _calculate_wave_defensive_pressure(self, minions: List[Dict], tower_pos: Tuple[float, float]) -> float:
        """计算兵线防守压力"""
        if not minions:
            return 0.0
        
        total_pressure = 0.0
        for minion in minions:
            minion_hp = float(minion.get('hp', 0))
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance = self._calculate_distance_by_pos(minion_position, tower_pos)
                
                if distance > 0:
                    pressure = minion_hp / max(distance, 100.0)
                    total_pressure += pressure
        
        return min(total_pressure / 1000.0, 1.0)
    
    def _count_minions_at_tower(self, minions: List[Dict], tower_pos: Tuple[float, float]) -> int:
        """计算在防御塔附近的小兵数量"""
        count = 0
        for minion in minions:
            minion_pos = minion.get('location', {})
            if minion_pos:
                minion_position = (float(minion_pos.get('x', 0)), float(minion_pos.get('z', 0)))
                distance = self._calculate_distance_by_pos(minion_position, tower_pos)
                if distance <= 1200:  # 在塔附近
                    count += 1
        return count
    
    def _update_state_history(self, current_state: Dict, frame_no: int):
        """更新状态历史"""
        history_entry = current_state.copy()
        history_entry['frame_no'] = frame_no
        history_entry['hero_gold'] = current_state.get('hero_gold', 0)
        
        self.state_history.append(history_entry)


class WaveStateTracker:
    """兵线状态追踪器"""
    
    def __init__(self):
        self.wave_history = deque(maxlen=20)
        self.last_wave_crash_frame = -1
        
    def update(self, current_state: Dict):
        """更新兵线状态"""
        self.wave_history.append(current_state)
    
    def detect_wave_crash_to_enemy_tower(self) -> bool:
        """检测送兵进塔事件"""
        if len(self.wave_history) < 2:
            return False
        
        current = self.wave_history[-1]
        previous = self.wave_history[-2]
        
        # 检查我方小兵到达敌方塔的事件
        current_minions = current.get('my_minions_at_enemy_tower', 0)
        previous_minions = previous.get('my_minions_at_enemy_tower', 0)
        
        # 如果小兵数量显著增加，说明有新的兵线到达
        if current_minions > previous_minions and current_minions >= 2:
            return True
        
        return False
    
    def evaluate_perfect_clear(self) -> float:
        """评估完美解线"""
        if len(self.wave_history) < 5:
            return 0.0
        
        # 检查敌方兵线是否被快速清理且我方塔血量损失很少
        current = self.wave_history[-1]
        past = self.wave_history[-5]
        
        enemy_minions_reduced = past.get('enemy_minions_at_my_tower', 0) - current.get('enemy_minions_at_my_tower', 0)
        tower_hp_loss = past.get('my_tower_hp', 1.0) - current.get('my_tower_hp', 1.0)
        
        if enemy_minions_reduced >= 2 and tower_hp_loss < 0.05:  # 清了兵且塔损失很少
            return min(enemy_minions_reduced / 3.0, 1.0)
        
        return 0.0
    
    def evaluate_wave_control(self) -> float:
        """评估兵线控制"""
        if len(self.wave_history) < 3:
            return 0.0
        
        # 简化评估：基于兵线优势的维持
        recent_advantages = [state.get('wave_advantage', 0.0) for state in list(self.wave_history)[-3:]]
        
        # 如果持续保持兵线优势
        if all(adv > 0.2 for adv in recent_advantages):
            return np.mean(recent_advantages)
        
        return 0.0


    def _calculate_victory_focused_rewards(self, victory_features: List[float], 
                                         current_state: Dict, main_hero: Dict, 
                                         enemy_hero: Dict, frame_data: Dict) -> Dict[str, float]:
        """计算胜利条件导向奖励 - 基于新增的5维特征"""
        rewards = {}
        
        if len(victory_features) < 5:
            return rewards
        
        # 解析胜利导向特征
        endgame_urgency = victory_features[0]          # 终局紧迫度
        one_push_potential = victory_features[1]       # 一波结束潜力
        defensive_desperation = victory_features[2]    # 防守绝境指数
        amplified_tower_diff = victory_features[3]     # 塔血量差异放大器
        decisive_timing = victory_features[4]          # 决战时机判断
        
        # 1. 终局紧迫状态奖励
        if endgame_urgency > 0.8:
            # 在终局阶段，奖励积极的行为
            if self._is_aggressive_action(frame_data):
                rewards['endgame_urgency_bonus'] = self.reward_weights['endgame_urgency_bonus'] * endgame_urgency
        
        # 2. 一波结束游戏奖励
        if one_push_potential > 0.6:
            # 当有一波结束的潜力时，奖励推塔行为
            tower_damage_dealt = self._get_tower_damage_dealt(frame_data, main_hero)
            if tower_damage_dealt > 0:
                rewards['one_push_victory_bonus'] = (
                    self.reward_weights['one_push_victory_bonus'] * 
                    one_push_potential * 
                    (tower_damage_dealt / 100.0)  # 根据伤害量调整
                )
        
        # 3. 绝境防守奖励
        if defensive_desperation > 0.7:
            # 在绝境状态下，奖励防守行为
            if self._is_defensive_action(frame_data, current_state):
                rewards['desperate_defense_bonus'] = (
                    self.reward_weights['desperate_defense_bonus'] * defensive_desperation
                )
            
            # 如果成功阻止敌方推塔，给予额外奖励
            my_tower_damage_taken = self._get_my_tower_damage_taken(frame_data)
            if my_tower_damage_taken == 0 and defensive_desperation > 0.8:
                rewards['critical_tower_save'] = self.reward_weights['critical_tower_save']
        
        # 4. 终局阶段塔血量优势放大奖励
        if abs(amplified_tower_diff) > 0.5:
            if amplified_tower_diff > 0:  # 我方塔血量优势
                rewards['tower_hp_amplified_bonus'] = (
                    self.reward_weights['tower_hp_amplified_bonus'] * 
                    amplified_tower_diff * 0.1  # 每帧的持续奖励
                )
            # 如果我方劣势，不给负奖励，让其他惩罚机制处理
        
        # 5. 决战时机把握奖励
        if abs(decisive_timing) > 0.6:
            if decisive_timing > 0:  # 应该主动决战
                if self._is_aggressive_action(frame_data):
                    rewards['decisive_timing_bonus'] = (
                        self.reward_weights['decisive_timing_bonus'] * decisive_timing
                    )
            else:  # 应该避免决战，寻求发育
                if self._is_farming_action(frame_data):
                    rewards['decisive_timing_bonus'] = (
                        self.reward_weights['decisive_timing_bonus'] * abs(decisive_timing) * 0.5
                    )
        
        # 6. 最后一击奖励
        enemy_tower_hp = current_state.get('enemy_tower_hp', 1.0)
        if enemy_tower_hp <= 0.1:  # 敌方塔血量极低
            tower_damage_dealt = self._get_tower_damage_dealt(frame_data, main_hero)
            if tower_damage_dealt > 0:
                # 越接近摧毁塔，奖励越高
                finishing_multiplier = (0.1 - enemy_tower_hp) / 0.1
                rewards['finishing_blow_bonus'] = (
                    self.reward_weights['finishing_blow_bonus'] * 
                    finishing_multiplier * 
                    (tower_damage_dealt / 50.0)
                )
        
        return rewards
    
    def _is_aggressive_action(self, frame_data: Dict) -> bool:
        """判断是否为积极进攻行为"""
        # 检查是否有对敌方英雄或防御塔的攻击行为
        for action in frame_data.get("frame_action", []):
            if action.get("type") in ["attack", "skill"]:
                target_type = action.get("target_type", "")
                if target_type in ["hero", "tower"]:
                    return True
        return False
    
    def _is_defensive_action(self, frame_data: Dict, current_state: Dict) -> bool:
        """判断是否为防守行为"""
        my_tower_pos = current_state.get('my_tower_pos', (0, 0))
        my_hero_pos = current_state.get('my_hero_pos', (0, 0))
        
        # 如果英雄靠近自己的塔，认为是防守行为
        if my_tower_pos and my_hero_pos:
            distance = ((my_hero_pos[0] - my_tower_pos[0])**2 + 
                       (my_hero_pos[1] - my_tower_pos[1])**2)**0.5
            return distance < 800  # 在塔附近800单位内
        return False
    
    def _is_farming_action(self, frame_data: Dict) -> bool:
        """判断是否为发育行为（攻击小兵）"""
        for action in frame_data.get("frame_action", []):
            if action.get("type") == "attack":
                target_type = action.get("target_type", "")
                if target_type == "minion":
                    return True
        return False
    
    def _get_tower_damage_dealt(self, frame_data: Dict, main_hero: Dict) -> float:
        """获取对敌方防御塔造成的伤害"""
        damage = 0.0
        for action in frame_data.get("frame_action", []):
            if (action.get("source_id") == main_hero.get("actor_id") and 
                action.get("target_type") == "tower" and
                action.get("target_camp") != main_hero.get("camp")):
                damage += action.get("damage", 0.0)
        return damage
    
    def _get_my_tower_damage_taken(self, frame_data: Dict) -> float:
        """获取我方防御塔受到的伤害"""
        damage = 0.0
        for action in frame_data.get("frame_action", []):
            if (action.get("target_type") == "tower" and 
                action.get("target_camp") == self.main_camp):
                damage += action.get("damage", 0.0)
        return damage


class StrategicDecisionAnalyzer:
    """战略决策分析器"""
    
    def __init__(self):
        self.decision_history = deque(maxlen=50)
    
    def analyze_decision_quality(self, action: str, state: Dict, outcome: Dict) -> float:
        """分析决策质量"""
        # 基于行动、状态和结果分析决策的合理性
        quality_score = 0.5  # 基础分数
        
        # 根据具体行动类型和状态进行评估
        if action == 'attack_tower':
            wave_advantage = state.get('wave_advantage', 0.0)
            if wave_advantage > 0.3:
                quality_score += 0.3  # 有兵线优势时攻击塔是好决策
            else:
                quality_score -= 0.2  # 无兵线优势时攻击塔风险大
        
        return max(0.0, min(quality_score, 1.0))
