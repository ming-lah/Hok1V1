#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Enhanced by Deep RL Analysis

兵线战略奖励系统
基于深度强化学习分析，实现完整的兵线奖励函数

核心理念：
1. 从"反应式"到"规划式"的奖励设计
2. 控线、慢推、快推重置的高级策略奖励
3. 经济收益、兵线控制、战略交互的完整建模
4. 势能奖励和事件奖励的结合
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict


class MinionStrategicRewards:
    """兵线战略奖励系统 - 实现完整的兵线奖励函数"""
    
    def __init__(self, main_hero_id: int, main_camp: str):
        self.main_hero_id = main_hero_id
        self.main_camp = main_camp
        self.enemy_camp = "PLAYERCAMP_2" if main_camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 历史状态追踪
        self.state_history = deque(maxlen=20)
        self.strategic_history = deque(maxlen=100)
        
        # 奖励权重配置
        self.reward_weights = {
            # 1. 经济收益奖励 (Economic Gain Rewards)
            'last_hit_gold': 1.0,               # 成功补刀金币奖励
            'last_hit_exp': 1.5,                # 成功补刀经验奖励 (前期更重要)
            'miss_last_hit_penalty': -0.5,      # 漏刀惩罚
            'perfect_last_hit_bonus': 2.0,      # 完美补刀奖励
            
            # 2. 兵线控制奖励 (Wave Control Rewards) - 势能奖励
            'wave_advantage_potential': 50.0,   # 兵线优势势能权重
            'freeze_sustain_bonus': 2.0,        # 稳定控线持续奖励 (每秒)
            'exp_deny_bonus': 5.0,              # 经验剥夺事件奖励
            'super_wave_formation': 100.0,      # 超级兵线形成奖励
            'wave_crash_bonus': 50.0,           # 兵线撞塔奖励
            'perfect_recall_bonus': 30.0,       # 完美回城奖励
            
            # 3. 战略交互奖励 (Strategic Interaction Rewards)
            'minion_aggro_penalty_multiplier': 2.0,  # 顶兵线战斗惩罚倍数
            'minion_shield_efficiency': 1.5,    # 利用小兵做盾牌的效率奖励
            'wave_timing_mastery': 20.0,        # 兵线时机掌控奖励
            
            # 4. 高级战略奖励 (Advanced Strategic Rewards)
            'slow_push_execution': 40.0,        # 慢推执行奖励
            'fast_push_reset': 25.0,            # 快推重置奖励
            'freeze_break_timing': 15.0,        # 破解控线时机奖励
            'lane_state_transition': 10.0,      # 兵线状态转换奖励
            
            # 5. 惩罚机制 (Penalty Mechanisms)
            'poor_wave_management': -20.0,      # 兵线管理不当惩罚
            'missed_farming_window': -15.0,     # 错失发育窗口惩罚
            'inefficient_recall': -25.0,        # 低效回城惩罚
            'strategic_blunder': -30.0,         # 战略失误惩罚
        }
        
        # 战略状态追踪
        self.strategic_tracker = StrategicStateTracker()
        
        # 势能函数历史
        self.potential_history = deque(maxlen=2)
        
        # 特殊事件检测器
        self.event_detector = MinionEventDetector(main_hero_id, main_camp)
        
    def calculate_minion_strategic_rewards(self, frame_data: Dict, main_hero: Dict, 
                                         enemy_hero: Dict, frame_no: int,
                                         minion_features: List[float] = None) -> Dict[str, float]:
        """计算兵线战略奖励"""
        if not main_hero or not enemy_hero:
            return {}
        
        rewards = {}
        
        # 提取当前兵线状态
        current_state = self._extract_current_minion_state(frame_data, main_hero, enemy_hero)
        
        # 1. 经济收益奖励
        economic_rewards = self._calculate_economic_gain_rewards(frame_data, main_hero, current_state)
        rewards.update(economic_rewards)
        
        # 2. 兵线控制奖励 (势能奖励)
        wave_control_rewards = self._calculate_wave_control_rewards(current_state, frame_no)
        rewards.update(wave_control_rewards)
        
        # 3. 战略交互奖励
        strategic_interaction_rewards = self._calculate_strategic_interaction_rewards(
            frame_data, main_hero, enemy_hero, current_state
        )
        rewards.update(strategic_interaction_rewards)
        
        # 4. 高级战略奖励
        advanced_strategic_rewards = self._calculate_advanced_strategic_rewards(
            current_state, frame_no
        )
        rewards.update(advanced_strategic_rewards)
        
        # 5. 惩罚机制
        penalty_rewards = self._calculate_penalty_rewards(frame_data, main_hero, current_state)
        rewards.update(penalty_rewards)
        
        # 更新历史状态
        self._update_strategic_history(current_state, frame_no)
        
        return rewards
    
    def _calculate_economic_gain_rewards(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算经济收益奖励 (基础能力)"""
        rewards = {}
        
        # 检测成功补刀
        last_hits = self.event_detector.detect_last_hits(frame_data, main_hero)
        
        for last_hit in last_hits:
            minion_type = last_hit['minion_type']
            gold_gained = last_hit['gold_gained']
            exp_gained = last_hit['exp_gained']
            
            # 成功补刀奖励
            rewards['last_hit_gold'] = rewards.get('last_hit_gold', 0) + gold_gained * self.reward_weights['last_hit_gold'] * 0.01
            rewards['last_hit_exp'] = rewards.get('last_hit_exp', 0) + exp_gained * self.reward_weights['last_hit_exp'] * 0.01
            
            # 特殊奖励：炮车补刀
            if minion_type == 'cannon':
                rewards['perfect_last_hit_bonus'] = self.reward_weights['perfect_last_hit_bonus']
        
        # 检测漏刀惩罚
        missed_last_hits = self.event_detector.detect_missed_last_hits(frame_data, main_hero, current_state)
        
        if missed_last_hits > 0:
            rewards['miss_last_hit_penalty'] = missed_last_hits * self.reward_weights['miss_last_hit_penalty']
        
        return rewards
    
    def _calculate_wave_control_rewards(self, current_state: Dict, frame_no: int) -> Dict[str, float]:
        """计算兵线控制奖励 (势能奖励)"""
        rewards = {}
        
        # 1. 兵线优势势能奖励 (基于您的势能函数思路)
        potential_rewards = self._calculate_wave_potential_rewards(current_state)
        rewards.update(potential_rewards)
        
        # 2. 稳定控线持续奖励
        freeze_rewards = self._calculate_freeze_rewards(current_state)
        rewards.update(freeze_rewards)
        
        # 3. 经验剥夺事件奖励
        exp_deny_rewards = self._calculate_exp_deny_rewards(current_state)
        rewards.update(exp_deny_rewards)
        
        # 4. 超级兵线形成奖励
        super_wave_rewards = self._calculate_super_wave_rewards(current_state)
        rewards.update(super_wave_rewards)
        
        # 5. 兵线撞塔奖励
        wave_crash_rewards = self._calculate_wave_crash_rewards(current_state, frame_no)
        rewards.update(wave_crash_rewards)
        
        return rewards
    
    def _calculate_strategic_interaction_rewards(self, frame_data: Dict, main_hero: Dict, 
                                               enemy_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算战略交互奖励"""
        rewards = {}
        
        # 1. 顶兵线战斗惩罚
        minion_aggro_penalty = self._calculate_minion_aggro_penalty(frame_data, main_hero, current_state)
        if minion_aggro_penalty < 0:
            rewards['minion_aggro_penalty'] = minion_aggro_penalty
        
        # 2. 利用小兵做盾牌奖励
        shield_efficiency = self._calculate_minion_shield_efficiency(frame_data, main_hero, current_state)
        if shield_efficiency > 0:
            rewards['minion_shield_efficiency'] = shield_efficiency
        
        # 3. 兵线时机掌控奖励
        timing_mastery = self._calculate_wave_timing_mastery(current_state, main_hero, enemy_hero)
        if timing_mastery > 0:
            rewards['wave_timing_mastery'] = timing_mastery
        
        return rewards
    
    def _calculate_advanced_strategic_rewards(self, current_state: Dict, frame_no: int) -> Dict[str, float]:
        """计算高级战略奖励"""
        rewards = {}
        
        if not self.state_history:
            return rewards
        
        prev_state = self.state_history[-1]
        
        # 1. 慢推执行奖励
        slow_push_reward = self._detect_slow_push_execution(current_state, prev_state)
        if slow_push_reward > 0:
            rewards['slow_push_execution'] = slow_push_reward
        
        # 2. 快推重置奖励
        fast_push_reward = self._detect_fast_push_reset(current_state, prev_state)
        if fast_push_reward > 0:
            rewards['fast_push_reset'] = fast_push_reward
        
        # 3. 破解控线时机奖励
        freeze_break_reward = self._detect_freeze_break_timing(current_state, prev_state)
        if freeze_break_reward > 0:
            rewards['freeze_break_timing'] = freeze_break_reward
        
        # 4. 兵线状态转换奖励
        transition_reward = self._detect_lane_state_transition(current_state, prev_state)
        if transition_reward > 0:
            rewards['lane_state_transition'] = transition_reward
        
        return rewards
    
    def _calculate_penalty_rewards(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算惩罚机制"""
        rewards = {}
        
        # 1. 兵线管理不当惩罚
        poor_management_penalty = self._detect_poor_wave_management(current_state)
        if poor_management_penalty < 0:
            rewards['poor_wave_management'] = poor_management_penalty
        
        # 2. 错失发育窗口惩罚
        missed_farming_penalty = self._detect_missed_farming_window(current_state)
        if missed_farming_penalty < 0:
            rewards['missed_farming_window'] = missed_farming_penalty
        
        # 3. 低效回城惩罚
        inefficient_recall_penalty = self._detect_inefficient_recall(frame_data, main_hero, current_state)
        if inefficient_recall_penalty < 0:
            rewards['inefficient_recall'] = inefficient_recall_penalty
        
        # 4. 战略失误惩罚
        strategic_blunder_penalty = self._detect_strategic_blunder(current_state)
        if strategic_blunder_penalty < 0:
            rewards['strategic_blunder'] = strategic_blunder_penalty
        
        return rewards
    
    def _calculate_wave_potential_rewards(self, current_state: Dict) -> Dict[str, float]:
        """计算兵线优势势能奖励 (基于您的势能函数设计)"""
        rewards = {}
        
        if not self.potential_history:
            # 首次计算，记录当前势能
            current_potential = self._calculate_wave_potential_function(current_state)
            self.potential_history.append(current_potential)
            return rewards
        
        # 计算势能变化 Φ(s_t) - Φ(s_{t-1})
        prev_potential = self.potential_history[-1]
        current_potential = self._calculate_wave_potential_function(current_state)
        
        potential_change = current_potential - prev_potential
        
        if abs(potential_change) > 0.01:  # 避免微小变化的噪音
            rewards['wave_advantage_potential'] = potential_change * self.reward_weights['wave_advantage_potential']
        
        # 更新势能历史
        self.potential_history.append(current_potential)
        
        return rewards
    
    def _calculate_freeze_rewards(self, current_state: Dict) -> Dict[str, float]:
        """计算控线奖励"""
        rewards = {}
        
        # 检查是否在控线区域 (我方塔前安全区)
        frontline_position = current_state.get('frontline_position', 0.0)
        enemy_in_exp_deny = current_state.get('enemy_in_exp_deny_position', False)
        
        # 控线条件：兵线在我方塔前安全区 + 敌人被压制
        if -0.4 < frontline_position < -0.1 and enemy_in_exp_deny:
            # 稳定控线持续奖励 (每帧给予小奖励)
            rewards['freeze_sustain_bonus'] = self.reward_weights['freeze_sustain_bonus'] / 30.0  # 每秒奖励除以30帧
        
        return rewards
    
    def _calculate_exp_deny_rewards(self, current_state: Dict) -> Dict[str, float]:
        """计算经验剥夺奖励"""
        rewards = {}
        
        # 检测经验剥夺事件
        if current_state.get('enemy_in_exp_deny_position', False):
            enemy_minion_deaths = current_state.get('enemy_minion_deaths_this_frame', 0)
            if enemy_minion_deaths > 0:
                # 敌方小兵死亡且敌方英雄在经验区外
                rewards['exp_deny_bonus'] = enemy_minion_deaths * self.reward_weights['exp_deny_bonus']
        
        return rewards
    
    def _calculate_super_wave_rewards(self, current_state: Dict) -> Dict[str, float]:
        """计算超级兵线奖励"""
        rewards = {}
        
        my_minion_count = current_state.get('my_minion_count', 0)
        frontline_position = current_state.get('frontline_position', 0.0)
        
        # 超级兵线形成条件：我方小兵数量>=8 且兵线过中线
        if my_minion_count >= 8 and frontline_position > 0.0:
            # 检查是否是新形成的超级兵线
            if not self.state_history or self.state_history[-1].get('my_minion_count', 0) < 8:
                rewards['super_wave_formation'] = self.reward_weights['super_wave_formation']
        
        return rewards
    
    def _calculate_wave_crash_rewards(self, current_state: Dict, frame_no: int) -> Dict[str, float]:
        """计算兵线撞塔奖励"""
        rewards = {}
        
        frontline_position = current_state.get('frontline_position', 0.0)
        
        # 兵线撞入敌方塔下 (前线位置 > 0.6)
        if frontline_position > 0.6:
            # 检查是否是新的撞塔事件
            if not self.state_history or self.state_history[-1].get('frontline_position', 0.0) <= 0.6:
                rewards['wave_crash_bonus'] = self.reward_weights['wave_crash_bonus']
                
                # 检查完美回城时机
                perfect_recall_reward = self._check_perfect_recall_timing(current_state, frame_no)
                if perfect_recall_reward > 0:
                    rewards['perfect_recall_bonus'] = perfect_recall_reward
        
        return rewards
    
    def _calculate_minion_aggro_penalty(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> float:
        """计算顶兵线战斗惩罚"""
        # 检查英雄受到的伤害
        hero_damage_received = self._get_hero_damage_received(frame_data, main_hero)
        
        if hero_damage_received > 0:
            # 计算攻击我方英雄的敌方小兵数量
            attacking_minions = current_state.get('enemy_minions_attacking_me', 0)
            
            if attacking_minions > 0:
                # 应用顶兵线战斗的惩罚倍数
                penalty_multiplier = 1 + (attacking_minions * 0.5 * self.reward_weights['minion_aggro_penalty_multiplier'])
                return -hero_damage_received * penalty_multiplier * 0.01  # 伤害值归一化
        
        return 0.0
    
    def _calculate_minion_shield_efficiency(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> float:
        """计算利用小兵做盾牌的效率"""
        # 这是一个间接奖励：当英雄在我方小兵身后时躲避了技能伤害
        # 简化实现：基于位置关系和未受到预期伤害
        
        my_minions_nearby = current_state.get('my_minions_nearby', 0)
        expected_damage = current_state.get('expected_enemy_damage', 0)
        actual_damage = self._get_hero_damage_received(frame_data, main_hero)
        
        if my_minions_nearby > 0 and expected_damage > actual_damage:
            # 有我方小兵在附近且实际受伤低于预期
            damage_avoided = expected_damage - actual_damage
            efficiency = damage_avoided * my_minions_nearby * self.reward_weights['minion_shield_efficiency'] * 0.01
            return min(efficiency, 10.0)  # 限制最大奖励
        
        return 0.0
    
    def _calculate_wave_timing_mastery(self, current_state: Dict, my_hero: Dict, enemy_hero: Dict) -> float:
        """计算兵线时机掌控奖励"""
        # 评估在正确时机做出的正确决策
        frontline_position = current_state.get('frontline_position', 0.0)
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        
        timing_score = 0.0
        
        # 兵线优势时主动进攻
        if frontline_position > 0.3 and my_hp_ratio > enemy_hp_ratio:
            timing_score += 0.5
        
        # 兵线劣势时保守发育
        if frontline_position < -0.3 and my_hp_ratio < enemy_hp_ratio:
            timing_score += 0.3
        
        # 兵线均势时合理换血
        if -0.2 < frontline_position < 0.2:
            timing_score += 0.2
        
        return timing_score * self.reward_weights['wave_timing_mastery']
    
    # ============ 高级战略检测方法 ============
    
    def _detect_slow_push_execution(self, current_state: Dict, prev_state: Dict) -> float:
        """检测慢推执行"""
        # 慢推的特征：我方小兵数量逐渐增加，兵线缓慢推进
        current_count = current_state.get('my_minion_count', 0)
        prev_count = prev_state.get('my_minion_count', 0)
        
        current_pos = current_state.get('frontline_position', 0.0)
        prev_pos = prev_state.get('frontline_position', 0.0)
        
        # 小兵数量增加且位置缓慢推进
        if current_count > prev_count and 0 < (current_pos - prev_pos) < 0.1:
            buildup_intensity = min((current_count - prev_count) / 2.0, 1.0)
            return buildup_intensity * self.reward_weights['slow_push_execution']
        
        return 0.0
    
    def _detect_fast_push_reset(self, current_state: Dict, prev_state: Dict) -> float:
        """检测快推重置"""
        # 快推的特征：兵线快速推进到敌方塔下，然后重置
        current_pos = current_state.get('frontline_position', 0.0)
        prev_pos = prev_state.get('frontline_position', 0.0)
        
        # 兵线从中线快速推到敌方塔下
        if prev_pos < 0.2 and current_pos > 0.6:
            push_speed = current_pos - prev_pos
            if push_speed > 0.3:  # 快速推进
                return min(push_speed * 2.0, 1.0) * self.reward_weights['fast_push_reset']
        
        return 0.0
    
    def _detect_freeze_break_timing(self, current_state: Dict, prev_state: Dict) -> float:
        """检测破解控线时机"""
        # 检测从控线状态转为推线的时机选择
        prev_pos = prev_state.get('frontline_position', 0.0)
        current_pos = current_state.get('frontline_position', 0.0)
        
        # 从控线区域 (-0.4, -0.1) 推出到中线
        if -0.4 < prev_pos < -0.1 and current_pos > 0.0:
            timing_quality = min((current_pos - prev_pos) / 0.5, 1.0)
            return timing_quality * self.reward_weights['freeze_break_timing']
        
        return 0.0
    
    def _detect_lane_state_transition(self, current_state: Dict, prev_state: Dict) -> float:
        """检测兵线状态转换"""
        # 奖励有意义的兵线状态转换
        prev_strategy = prev_state.get('strategic_state', {}).get('current_strategy', 'unknown')
        current_strategy = current_state.get('strategic_state', {}).get('current_strategy', 'unknown')
        
        # 有效的状态转换
        valid_transitions = {
            ('neutral', 'freezing'): 0.6,
            ('freezing', 'slow_pushing'): 0.8,
            ('slow_pushing', 'fast_pushing'): 1.0,
            ('fast_pushing', 'neutral'): 0.4,
        }
        
        transition = (prev_strategy, current_strategy)
        if transition in valid_transitions:
            transition_value = valid_transitions[transition]
            return transition_value * self.reward_weights['lane_state_transition']
        
        return 0.0
    
    # ============ 惩罚检测方法 ============
    
    def _detect_poor_wave_management(self, current_state: Dict) -> float:
        """检测兵线管理不当"""
        # 检测明显的兵线管理错误
        frontline_position = current_state.get('frontline_position', 0.0)
        my_minion_count = current_state.get('my_minion_count', 0)
        enemy_minion_count = current_state.get('enemy_minion_count', 0)
        
        penalty = 0.0
        
        # 兵线严重劣势时没有及时调整
        if frontline_position < -0.7 and enemy_minion_count > my_minion_count + 3:
            penalty += 0.5
        
        # 兵线优势时没有利用
        if frontline_position > 0.5 and my_minion_count < enemy_minion_count:
            penalty += 0.3
        
        return -penalty * self.reward_weights['poor_wave_management'] if penalty > 0 else 0.0
    
    def _detect_missed_farming_window(self, current_state: Dict) -> float:
        """检测错失发育窗口"""
        # 检测有明显发育机会但没有把握的情况
        last_hitable_gold = current_state.get('last_hitable_gold', 0)
        last_hits_taken = current_state.get('last_hits_taken_this_frame', 0)
        
        # 有很多可补刀的小兵但没有补到
        if last_hitable_gold > 100 and last_hits_taken == 0:
            missed_opportunity = min(last_hitable_gold / 200.0, 1.0)
            return -missed_opportunity * self.reward_weights['missed_farming_window']
        
        return 0.0
    
    def _detect_inefficient_recall(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> float:
        """检测低效回城"""
        # 检查英雄是否在不合适的时机回城
        is_recalling = self._is_hero_recalling(frame_data, main_hero)
        
        if is_recalling:
            frontline_position = current_state.get('frontline_position', 0.0)
            enemy_minions_attacking_tower = current_state.get('enemy_minions_attacking_my_tower', 0)
            
            # 在敌方兵线攻击我方塔时回城
            if frontline_position < -0.5 and enemy_minions_attacking_tower > 0:
                return self.reward_weights['inefficient_recall']
        
        return 0.0
    
    def _detect_strategic_blunder(self, current_state: Dict) -> float:
        """检测战略失误"""
        # 检测严重的战略决策失误
        frontline_position = current_state.get('frontline_position', 0.0)
        my_minion_count = current_state.get('my_minion_count', 0)
        
        # 有大波兵线但让其白白撞塔消失
        if my_minion_count >= 8 and frontline_position > 0.8:
            # 超级兵线撞塔但没有跟进
            hero_near_tower = current_state.get('hero_near_enemy_tower', False)
            if not hero_near_tower:
                return self.reward_weights['strategic_blunder']
        
        return 0.0
    
    # ============ 辅助方法 ============
    
    def _calculate_wave_potential_function(self, state: Dict) -> float:
        """计算兵线势能函数 Φ(s)"""
        # 基于您的势能函数设计思路
        wave_advantage = state.get('wave_advantage', 0.0)
        frontline_position = state.get('frontline_position', 0.0)
        
        # 势能 = 兵线优势度 + 位置优势
        potential = 0.7 * wave_advantage + 0.3 * frontline_position
        return potential
    
    def _extract_current_minion_state(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict:
        """提取当前兵线状态"""
        minion_states = self._extract_minion_states(frame_data.get("npc_states", []))
        my_minions = minion_states.get('my_minions', [])
        enemy_minions = minion_states.get('enemy_minions', [])
        
        # 计算基础状态
        frontline_position = self._calculate_frontline_position(my_minions, enemy_minions)
        wave_advantage = self._calculate_wave_advantage(my_minions, enemy_minions)
        
        return {
            'minion_states': minion_states,
            'my_minions': my_minions,
            'enemy_minions': enemy_minions,
            'my_minion_count': len(my_minions),
            'enemy_minion_count': len(enemy_minions),
            'frontline_position': frontline_position,
            'wave_advantage': wave_advantage,
            'enemy_in_exp_deny_position': self._is_enemy_in_exp_deny_position(enemy_hero, enemy_minions),
            'last_hitable_gold': self._calculate_last_hitable_gold(enemy_minions, main_hero),
            'strategic_state': {'current_strategy': 'neutral'},  # 简化处理
        }
    
    def _check_perfect_recall_timing(self, current_state: Dict, frame_no: int) -> float:
        """检查完美回城时机"""
        # 简化实现：兵线刚撞入敌方塔下是回城的好时机
        frontline_position = current_state.get('frontline_position', 0.0)
        
        if frontline_position > 0.6:
            return self.reward_weights['perfect_recall_bonus']
        
        return 0.0
    
    def _get_hero_damage_received(self, frame_data: Dict, hero: Dict) -> float:
        """获取英雄受到的伤害"""
        hero_id = hero.get("player_id", -1)
        total_damage = 0.0
        
        for action in frame_data.get("frame_action", []):
            hurt_action = action.get("hurt_action", {})
            if hurt_action:
                target_player_id = hurt_action.get("target_player_id", -1)
                damage = float(hurt_action.get("damage", 0))
                
                if target_player_id == hero_id:
                    total_damage += damage
        
        return total_damage
    
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
    
    def _is_hero_recalling(self, frame_data: Dict, hero: Dict) -> bool:
        """判断英雄是否在回城"""
        # 简化实现：检查特定的回城动作
        return False  # 需要根据实际数据协议实现
    
    # ============ 复用已有方法 ============
    
    def _extract_minion_states(self, npcs: List) -> Dict:
        """提取小兵状态"""
        minion_states = {'my_minions': [], 'enemy_minions': []}
        
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
    
    def _calculate_wave_advantage(self, my_minions: List[Dict], enemy_minions: List[Dict]) -> float:
        """计算兵线优势"""
        my_hp = sum(float(m.get('hp', 0)) for m in my_minions)
        enemy_hp = sum(float(m.get('hp', 0)) for m in enemy_minions)
        
        if my_hp + enemy_hp == 0:
            return 0.0
        
        return (my_hp - enemy_hp) / (my_hp + enemy_hp)
    
    def _is_enemy_in_exp_deny_position(self, enemy_hero: Dict, enemy_minions: List[Dict]) -> bool:
        """判断敌方英雄是否处于被经验压制的位置"""
        # 简化实现
        return False
    
    def _calculate_last_hitable_gold(self, enemy_minions: List[Dict], my_hero: Dict) -> float:
        """计算可补刀的金币价值"""
        # 简化实现
        return 0.0
    
    def _update_strategic_history(self, current_state: Dict, frame_no: int):
        """更新战略历史"""
        history_entry = current_state.copy()
        history_entry['frame_no'] = frame_no
        
        self.state_history.append(history_entry)


class StrategicStateTracker:
    """战略状态追踪器"""
    
    def __init__(self):
        self.current_strategy = 'neutral'
        self.strategy_duration = 0
        
    def update_strategy(self, minion_state: Dict):
        """更新战略状态"""
        # 基于兵线状态判断当前策略
        frontline_pos = minion_state.get('frontline_position', 0.0)
        my_count = minion_state.get('my_minion_count', 0)
        
        if -0.4 < frontline_pos < -0.1:
            self.current_strategy = 'freezing'
        elif my_count >= 8:
            self.current_strategy = 'slow_pushing'
        elif frontline_pos > 0.6:
            self.current_strategy = 'fast_pushing'
        else:
            self.current_strategy = 'neutral'


class MinionEventDetector:
    """小兵事件检测器"""
    
    def __init__(self, main_hero_id: int, main_camp: str):
        self.main_hero_id = main_hero_id
        self.main_camp = main_camp
        
    def detect_last_hits(self, frame_data: Dict, main_hero: Dict) -> List[Dict]:
        """检测成功补刀事件"""
        last_hits = []
        
        for action in frame_data.get("frame_action", []):
            dead_action = action.get("dead_action", {})
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            
            if ("SOLDIER" in str(death.get("type", "")) and 
                killer.get("player_id") == self.main_hero_id):
                
                # 判断小兵类型
                minion_type = 'melee'  # 简化处理
                if 'CANNON' in str(death.get("type", "")):
                    minion_type = 'cannon'
                elif 'REMOTE' in str(death.get("type", "")):
                    minion_type = 'ranged'
                
                last_hits.append({
                    'minion_type': minion_type,
                    'gold_gained': 20,  # 简化处理
                    'exp_gained': 30    # 简化处理
                })
        
        return last_hits
    
    def detect_missed_last_hits(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> int:
        """检测漏刀事件"""
        # 简化实现：计算附近死亡的敌方小兵但不是我方击杀的
        missed_count = 0
        
        for action in frame_data.get("frame_action", []):
            dead_action = action.get("dead_action", {})
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            
            if ("SOLDIER" in str(death.get("type", "")) and 
                death.get("camp") != self.main_camp and
                killer.get("player_id") != self.main_hero_id):
                # 敌方小兵死亡但不是我方击杀 - 可能是漏刀
                missed_count += 1
        
        return missed_count
