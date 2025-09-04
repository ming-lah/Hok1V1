#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Enhanced by Deep RL Analysis

稠密奖励函数处理器
基于深度强化学习分析，实现完整的奖励塑形系统

设计原则：
1. 稠密性：每个时间步都有奖励信号
2. 分层性：将最终目标分解为多个子目标
3. 平衡性：避免某一项奖励过度主导
4. 可调性：所有权重都可以灵活调整
5. 可解释性：每项奖励都有明确的意义
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict


class DenseRewardProcessor:
    """稠密奖励处理器 - 实现完整的奖励塑形系统"""
    
    def __init__(self, main_hero_id: int, main_camp: str):
        self.main_hero_id = main_hero_id
        self.main_camp = main_camp
        self.enemy_camp = "PLAYERCAMP_2" if main_camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 历史状态追踪
        self.state_history = deque(maxlen=10)
        self.reward_history = deque(maxlen=100)
        
        # 奖励权重配置 - 基于您的深度分析设计
        self.reward_weights = {
            # 1. 核心目标奖励 (Primary Objective Rewards)
            'game_victory': 1000.0,         # 赢得比赛
            'game_defeat': -1000.0,         # 输掉比赛
            'tower_destroy': 1000.0,        # 摧毁敌方防御塔
            'tower_destroyed': -1000.0,     # 我方防御塔被摧毁
            
            # 2. 推塔相关奖励 (Tower Damage Rewards)
            'tower_damage_dealt': 2.0,      # 对防御塔造成伤害 (每点伤害)
            'tower_damage_received': -2.0,  # 我方防御塔受到伤害
            'tower_hp_advantage': 5.0,      # 防御塔血量优势维持
            
            # 3. 英雄对战奖励 (Hero Combat Rewards)
            'hero_damage_dealt': 1.0,       # 对敌方英雄造成伤害
            'hero_damage_received': -1.0,   # 受到敌方英雄伤害
            'hero_kill': 200.0,             # 击杀敌方英雄
            'hero_death': -200.0,           # 自己死亡
            'hp_advantage_maintain': 1.0,   # 维持血量优势
            
            # 4. 经济发育奖励 (Economic Development Rewards)
            'gold_gain': 0.02,              # 金币获取 (每金币)
            'exp_gain': 0.01,               # 经验获取 (每点经验)
            'last_hit_success': 5.0,        # 成功补刀
            'last_hit_miss': -2.0,          # 错失补刀
            'economic_advantage': 2.0,      # 经济优势维持
            'farming_efficiency': 3.0,      # 发育效率
            
            # 5. 生存与血量管理 (Survival & HP Management)
            'hp_ratio_maintain': 1.0,       # 维持健康血量
            'low_hp_penalty': -2.0,         # 低血量惩罚
            'hp_recovery': 2.0,             # 血量回复奖励
            'safe_positioning': 1.5,        # 安全位置奖励
            
            # 6. 技能使用奖励 (Skill Usage Rewards)
            'skill_hit_hero': 3.0,          # 技能命中英雄
            'skill_miss': -1.0,             # 技能未命中
            'combo_execution': 10.0,        # 连招执行
            'skill_cd_management': 1.0,     # 技能冷却管理
            
            # 7. 位置与移动奖励 (Positioning & Movement Rewards)
            'position_advantage': 2.0,      # 位置优势
            'safe_retreat': 3.0,            # 安全撤退
            'aggressive_advance': 2.0,      # 主动进攻
            'movement_efficiency': 1.0,     # 移动效率
            
            # 8. 战术执行奖励 (Tactical Execution Rewards)
            'economic_pressure': 4.0,       # 经济压制执行
            'defensive_farming': 3.0,       # 防守发育
            'tempo_control': 2.0,           # 节奏控制
            'resource_control': 2.0,        # 资源控制
            
            # 9. 时序奖励 (Temporal Rewards)
            'early_game_farming': 2.0,      # 前期发育
            'mid_game_fighting': 3.0,       # 中期对战
            'late_game_finishing': 4.0,     # 后期终结
            'game_pace_control': 1.5,       # 游戏节奏控制
            
            # 10. 惩罚机制 (Penalty Mechanisms)
            'inefficient_action': -1.0,     # 无效行动
            'resource_waste': -2.0,         # 资源浪费
            'tactical_error': -3.0,         # 战术错误
            'positioning_error': -2.0,      # 位置错误
        }
        
        # 动态权重调整因子
        self.dynamic_weights = {
            'aggression_factor': 1.0,       # 攻击性因子
            'safety_factor': 1.0,           # 安全性因子
            'economic_factor': 1.0,         # 经济性因子
            'tactical_factor': 1.0          # 战术性因子
        }
        
        # 阈值配置
        self.thresholds = {
            'low_hp_threshold': 0.3,        # 低血量阈值
            'high_hp_threshold': 0.8,       # 高血量阈值
            'economic_advantage_threshold': 500,  # 经济优势阈值
            'safe_distance_threshold': 800,  # 安全距离阈值
            'attack_distance_threshold': 600, # 攻击距离阈值
        }
    
    def calculate_dense_rewards(self, frame_data: Dict, main_hero: Dict, 
                              enemy_hero: Dict, frame_no: int) -> Dict[str, float]:
        """计算稠密奖励 - 每个时间步的完整奖励"""
        if not main_hero or not enemy_hero:
            return {}
        
        rewards = {}
        
        # 1. 核心目标奖励
        core_rewards = self._calculate_core_objective_rewards(frame_data, main_hero, enemy_hero)
        rewards.update(core_rewards)
        
        # 2. 推塔相关奖励
        tower_rewards = self._calculate_tower_rewards(frame_data, main_hero, enemy_hero)
        rewards.update(tower_rewards)
        
        # 3. 英雄对战奖励
        combat_rewards = self._calculate_combat_rewards(frame_data, main_hero, enemy_hero)
        rewards.update(combat_rewards)
        
        # 4. 经济发育奖励
        economic_rewards = self._calculate_economic_rewards(frame_data, main_hero, enemy_hero)
        rewards.update(economic_rewards)
        
        # 5. 生存与血量管理奖励
        survival_rewards = self._calculate_survival_rewards(main_hero, enemy_hero)
        rewards.update(survival_rewards)
        
        # 6. 技能使用奖励
        skill_rewards = self._calculate_skill_rewards(frame_data, main_hero, enemy_hero)
        rewards.update(skill_rewards)
        
        # 7. 位置与移动奖励
        positioning_rewards = self._calculate_positioning_rewards(main_hero, enemy_hero, frame_data)
        rewards.update(positioning_rewards)
        
        # 8. 战术执行奖励 (基于您的零和博弈思路)
        tactical_rewards = self._calculate_tactical_rewards(main_hero, enemy_hero, frame_data)
        rewards.update(tactical_rewards)
        
        # 9. 时序奖励
        temporal_rewards = self._calculate_temporal_rewards(main_hero, enemy_hero, frame_no)
        rewards.update(temporal_rewards)
        
        # 10. 惩罚机制
        penalty_rewards = self._calculate_penalty_rewards(main_hero, enemy_hero, frame_data)
        rewards.update(penalty_rewards)
        
        # 应用动态权重调整
        adjusted_rewards = self._apply_dynamic_weights(rewards, main_hero, enemy_hero)
        
        # 更新状态历史
        self._update_reward_history(main_hero, enemy_hero, frame_data, frame_no)
        
        return adjusted_rewards
    
    def _calculate_core_objective_rewards(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict[str, float]:
        """计算核心目标奖励"""
        rewards = {}
        
        # 检查胜负
        if main_hero.get("win", False):
            rewards['game_victory'] = self.reward_weights['game_victory']
        elif enemy_hero.get("win", False):
            rewards['game_defeat'] = self.reward_weights['game_defeat']
        
        # 检查塔的摧毁 - 通过frame_action检测
        for action in frame_data.get("frame_action", []):
            dead_action = action.get("dead_action", {})
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            
            if "TOWER" in str(death.get("type", "")):
                if death.get("camp") != self.main_camp:  # 摧毁敌方塔
                    rewards['tower_destroy'] = self.reward_weights['tower_destroy']
                else:  # 我方塔被摧毁
                    rewards['tower_destroyed'] = self.reward_weights['tower_destroyed']
        
        return rewards
    
    def _calculate_tower_rewards(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict[str, float]:
        """计算推塔相关奖励"""
        rewards = {}
        
        if not self.state_history:
            return rewards
        
        prev_state = self.state_history[-1]
        
        # 计算防御塔血量变化
        current_tower_hp = self._get_tower_hp_info(frame_data)
        prev_tower_hp = prev_state.get('tower_hp', current_tower_hp)
        
        # 对敌方塔造成伤害
        enemy_tower_damage = prev_tower_hp.get('enemy', 1.0) - current_tower_hp.get('enemy', 1.0)
        if enemy_tower_damage > 0:
            rewards['tower_damage_dealt'] = enemy_tower_damage * self.reward_weights['tower_damage_dealt']
        
        # 我方塔受到伤害
        my_tower_damage = prev_tower_hp.get('my', 1.0) - current_tower_hp.get('my', 1.0)
        if my_tower_damage > 0:
            rewards['tower_damage_received'] = my_tower_damage * self.reward_weights['tower_damage_received']
        
        # 防御塔血量优势维持
        tower_advantage = current_tower_hp.get('my', 0.0) - current_tower_hp.get('enemy', 0.0)
        if tower_advantage > 0:
            rewards['tower_hp_advantage'] = tower_advantage * self.reward_weights['tower_hp_advantage'] * 0.1
        
        return rewards
    
    def _calculate_combat_rewards(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict[str, float]:
        """计算英雄对战奖励"""
        rewards = {}
        
        # 检查击杀和死亡
        for action in frame_data.get("frame_action", []):
            dead_action = action.get("dead_action", {})
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            
            if death.get("player_id") == self.main_hero_id:  # 我方死亡
                rewards['hero_death'] = self.reward_weights['hero_death']
            elif killer.get("player_id") == self.main_hero_id:  # 我方击杀
                rewards['hero_kill'] = self.reward_weights['hero_kill']
        
        # 计算伤害奖励
        if self.state_history:
            prev_state = self.state_history[-1]
            
            # 我方对敌方造成的伤害
            current_damage_dealt = float(main_hero.get("totalHurtToHero", 0))
            prev_damage_dealt = prev_state.get('damage_dealt', current_damage_dealt)
            damage_increase = current_damage_dealt - prev_damage_dealt
            if damage_increase > 0:
                rewards['hero_damage_dealt'] = damage_increase * self.reward_weights['hero_damage_dealt'] * 0.01
            
            # 我方受到的伤害
            current_damage_received = float(main_hero.get("totalBeHurtByHero", 0))
            prev_damage_received = prev_state.get('damage_received', current_damage_received)
            damage_received_increase = current_damage_received - prev_damage_received
            if damage_received_increase > 0:
                rewards['hero_damage_received'] = damage_received_increase * self.reward_weights['hero_damage_received'] * 0.01
        
        # 血量优势维持
        my_hp_ratio = self._get_hp_ratio(main_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        hp_advantage = my_hp_ratio - enemy_hp_ratio
        if hp_advantage > 0:
            rewards['hp_advantage_maintain'] = hp_advantage * self.reward_weights['hp_advantage_maintain'] * 0.1
        
        return rewards
    
    def _calculate_economic_rewards(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict[str, float]:
        """计算经济发育奖励"""
        rewards = {}
        
        if not self.state_history:
            return rewards
        
        prev_state = self.state_history[-1]
        
        # 金币获取奖励
        current_gold = float(main_hero.get("money", 0))
        prev_gold = prev_state.get('my_gold', current_gold)
        gold_gain = current_gold - prev_gold
        if gold_gain > 0:
            rewards['gold_gain'] = gold_gain * self.reward_weights['gold_gain']
        
        # 经验获取奖励
        current_exp = float(main_hero.get("exp", 0))
        prev_exp = prev_state.get('my_exp', current_exp)
        exp_gain = current_exp - prev_exp
        if exp_gain > 0:
            rewards['exp_gain'] = exp_gain * self.reward_weights['exp_gain']
        
        # 补刀奖励
        last_hit_count = self._count_last_hits(frame_data)
        if last_hit_count > 0:
            rewards['last_hit_success'] = last_hit_count * self.reward_weights['last_hit_success']
        
        # 经济优势奖励
        enemy_gold = float(enemy_hero.get("money", 0))
        economic_advantage = current_gold - enemy_gold
        if economic_advantage > self.thresholds['economic_advantage_threshold']:
            advantage_ratio = min(economic_advantage / 2000.0, 2.0)
            rewards['economic_advantage'] = advantage_ratio * self.reward_weights['economic_advantage'] * 0.1
        
        # 发育效率奖励 (金币获取与风险的比例)
        if gold_gain > 0:
            # 计算发育期间的风险 (血量损失)
            hp_loss = max(0, prev_state.get('my_hp_ratio', 1.0) - self._get_hp_ratio(main_hero))
            risk_factor = max(hp_loss, 0.01)  # 避免除零
            efficiency = gold_gain / (risk_factor * 100)
            rewards['farming_efficiency'] = min(efficiency, 5.0) * self.reward_weights['farming_efficiency'] * 0.1
        
        return rewards
    
    def _calculate_survival_rewards(self, main_hero: Dict, enemy_hero: Dict) -> Dict[str, float]:
        """计算生存与血量管理奖励"""
        rewards = {}
        
        my_hp_ratio = self._get_hp_ratio(main_hero)
        
        # 维持健康血量奖励
        if my_hp_ratio > self.thresholds['high_hp_threshold']:
            rewards['hp_ratio_maintain'] = (my_hp_ratio - self.thresholds['high_hp_threshold']) * \
                                         self.reward_weights['hp_ratio_maintain']
        
        # 低血量惩罚
        if my_hp_ratio < self.thresholds['low_hp_threshold']:
            penalty_intensity = (self.thresholds['low_hp_threshold'] - my_hp_ratio) / self.thresholds['low_hp_threshold']
            rewards['low_hp_penalty'] = penalty_intensity * self.reward_weights['low_hp_penalty']
        
        # 血量回复奖励
        if self.state_history:
            prev_hp = self.state_history[-1].get('my_hp_ratio', my_hp_ratio)
            hp_recovery = my_hp_ratio - prev_hp
            if hp_recovery > 0:
                rewards['hp_recovery'] = hp_recovery * self.reward_weights['hp_recovery']
        
        # 安全位置奖励
        distance = self._calculate_distance(main_hero, enemy_hero)
        if distance > self.thresholds['safe_distance_threshold'] and my_hp_ratio < 0.6:
            safety_bonus = min((distance - self.thresholds['safe_distance_threshold']) / 500.0, 1.0)
            rewards['safe_positioning'] = safety_bonus * self.reward_weights['safe_positioning'] * 0.1
        
        return rewards
    
    def _calculate_skill_rewards(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict[str, float]:
        """计算技能使用奖励"""
        rewards = {}
        
        # 技能命中奖励 - 基于技能使用统计
        skill_state = main_hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        
        if self.state_history:
            prev_state = self.state_history[-1]
            
            for i, slot in enumerate(slots):
                current_hit = int(slot.get("hitHeroTimes", 0))
                prev_hit = prev_state.get(f'skill_{i}_hit', current_hit)
                
                if current_hit > prev_hit:
                    # 技能命中英雄
                    rewards['skill_hit_hero'] = self.reward_weights['skill_hit_hero']
                
                current_used = int(slot.get("usedTimes", 0))
                prev_used = prev_state.get(f'skill_{i}_used', current_used)
                
                if current_used > prev_used and current_hit == prev_hit:
                    # 技能使用了但没命中
                    rewards['skill_miss'] = self.reward_weights['skill_miss']
        
        # 连招执行奖励 (简化检测)
        combo_reward = self._detect_combo_execution(main_hero)
        if combo_reward > 0:
            rewards['combo_execution'] = combo_reward * self.reward_weights['combo_execution']
        
        return rewards
    
    def _calculate_positioning_rewards(self, main_hero: Dict, enemy_hero: Dict, frame_data: Dict) -> Dict[str, float]:
        """计算位置与移动奖励"""
        rewards = {}
        
        distance = self._calculate_distance(main_hero, enemy_hero)
        my_hp_ratio = self._get_hp_ratio(main_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        
        # 位置优势奖励
        position_advantage = self._calculate_position_advantage(main_hero, enemy_hero, frame_data)
        if position_advantage > 0:
            rewards['position_advantage'] = position_advantage * self.reward_weights['position_advantage'] * 0.1
        
        # 安全撤退奖励
        if self.state_history:
            prev_distance = self.state_history[-1].get('distance', distance)
            distance_change = distance - prev_distance
            
            # 低血量时拉开距离
            if my_hp_ratio < enemy_hp_ratio and distance_change > 0:
                retreat_quality = min(distance_change / 200.0, 1.0) * (enemy_hp_ratio - my_hp_ratio)
                rewards['safe_retreat'] = retreat_quality * self.reward_weights['safe_retreat']
            
            # 血量优势时主动进攻
            elif my_hp_ratio > enemy_hp_ratio + 0.2 and distance_change < 0:
                advance_quality = min(abs(distance_change) / 200.0, 1.0) * (my_hp_ratio - enemy_hp_ratio)
                rewards['aggressive_advance'] = advance_quality * self.reward_weights['aggressive_advance']
        
        return rewards
    
    def _calculate_tactical_rewards(self, main_hero: Dict, enemy_hero: Dict, frame_data: Dict) -> Dict[str, float]:
        """计算战术执行奖励 - 基于您的零和博弈思路"""
        rewards = {}
        
        my_money = float(main_hero.get("money", 0))
        enemy_money = float(enemy_hero.get("money", 0))
        money_delta = my_money - enemy_money
        distance = self._calculate_distance(main_hero, enemy_hero)
        actor_state = main_hero.get("actor_state", {})
        if isinstance(actor_state, dict):
            attack_range = float(actor_state.get("attack_range", 600))
        else:
            attack_range = 600.0
        
        # 经济压制执行 (经济优势时的正确行为)
        if money_delta > 0:
            if distance <= attack_range:
                # 经济优势且在攻击范围内，应该攻击英雄
                hero_attack_action = self._detect_hero_attack_action(frame_data, main_hero)
                if hero_attack_action:
                    pressure_intensity = min(money_delta / 1000.0, 2.0)
                    rewards['economic_pressure'] = pressure_intensity * self.reward_weights['economic_pressure'] * 0.1
            else:
                # 经济优势但不在攻击范围，应该发育或追击
                farming_action = self._detect_farming_action(frame_data)
                if farming_action:
                    rewards['economic_pressure'] = 0.5 * self.reward_weights['economic_pressure'] * 0.1
        
        # 防守发育 (经济劣势时的正确行为)
        elif money_delta < -500:
            farming_action = self._detect_farming_action(frame_data)
            safe_distance = distance > 800
            if farming_action and safe_distance:
                defensive_quality = min(abs(money_delta) / 1000.0, 2.0)
                rewards['defensive_farming'] = defensive_quality * self.reward_weights['defensive_farming'] * 0.1
        
        # 节奏控制奖励
        tempo_control = self._assess_tempo_control(main_hero, enemy_hero, frame_data)
        if tempo_control > 0:
            rewards['tempo_control'] = tempo_control * self.reward_weights['tempo_control'] * 0.1
        
        return rewards
    
    def _calculate_temporal_rewards(self, main_hero: Dict, enemy_hero: Dict, frame_no: int) -> Dict[str, float]:
        """计算时序奖励"""
        rewards = {}
        
        game_progress = min(frame_no / 18000.0, 1.0)  # 10分钟游戏
        
        # 前期发育奖励
        if game_progress < 0.3:
            farming_quality = self._assess_farming_quality(main_hero)
            rewards['early_game_farming'] = farming_quality * self.reward_weights['early_game_farming'] * 0.1
        
        # 中期对战奖励
        elif 0.3 <= game_progress < 0.7:
            fighting_quality = self._assess_fighting_quality(main_hero, enemy_hero)
            rewards['mid_game_fighting'] = fighting_quality * self.reward_weights['mid_game_fighting'] * 0.1
        
        # 后期终结奖励
        else:
            finishing_quality = self._assess_finishing_quality(main_hero, enemy_hero)
            rewards['late_game_finishing'] = finishing_quality * self.reward_weights['late_game_finishing'] * 0.1
        
        return rewards
    
    def _calculate_penalty_rewards(self, main_hero: Dict, enemy_hero: Dict, frame_data: Dict) -> Dict[str, float]:
        """计算惩罚机制"""
        rewards = {}
        
        # 无效行动惩罚 (例如：在安全距离外使用近程技能)
        inefficient_actions = self._detect_inefficient_actions(main_hero, enemy_hero, frame_data)
        if inefficient_actions > 0:
            rewards['inefficient_action'] = inefficient_actions * self.reward_weights['inefficient_action']
        
        # 资源浪费惩罚 (例如：满蓝时不使用技能，错失补刀机会)
        resource_waste = self._detect_resource_waste(main_hero, frame_data)
        if resource_waste > 0:
            rewards['resource_waste'] = resource_waste * self.reward_weights['resource_waste']
        
        # 位置错误惩罚 (例如：低血量时过于接近敌人)
        positioning_errors = self._detect_positioning_errors(main_hero, enemy_hero)
        if positioning_errors > 0:
            rewards['positioning_error'] = positioning_errors * self.reward_weights['positioning_error']
        
        return rewards
    
    def _apply_dynamic_weights(self, rewards: Dict[str, float], main_hero: Dict, enemy_hero: Dict) -> Dict[str, float]:
        """应用动态权重调整"""
        adjusted_rewards = {}
        
        # 根据游戏状态调整权重
        my_hp_ratio = self._get_hp_ratio(main_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        my_money = float(main_hero.get("money", 0))
        enemy_money = float(enemy_hero.get("money", 0))
        
        # 动态调整因子
        if my_hp_ratio < 0.3:
            self.dynamic_weights['safety_factor'] = 2.0
            self.dynamic_weights['aggression_factor'] = 0.5
        elif my_hp_ratio > 0.8 and my_money > enemy_money:
            self.dynamic_weights['aggression_factor'] = 1.5
            self.dynamic_weights['safety_factor'] = 0.8
        
        # 应用权重调整
        for reward_type, reward_value in rewards.items():
            if 'damage_dealt' in reward_type or 'kill' in reward_type or 'pressure' in reward_type:
                adjusted_rewards[reward_type] = reward_value * self.dynamic_weights['aggression_factor']
            elif 'retreat' in reward_type or 'safe' in reward_type or 'hp' in reward_type:
                adjusted_rewards[reward_type] = reward_value * self.dynamic_weights['safety_factor']
            elif 'gold' in reward_type or 'farming' in reward_type or 'economic' in reward_type:
                adjusted_rewards[reward_type] = reward_value * self.dynamic_weights['economic_factor']
            else:
                adjusted_rewards[reward_type] = reward_value
        
        return adjusted_rewards
    
    # ============ 辅助方法 ============
    
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
        """计算距离"""
        if not isinstance(hero1, dict) or not isinstance(hero2, dict):
            return 0.0
        
        actor_state1 = hero1.get("actor_state", {})
        actor_state2 = hero2.get("actor_state", {})
        
        if not isinstance(actor_state1, dict) or not isinstance(actor_state2, dict):
            return 0.0
        
        pos1 = actor_state1.get("location", {})
        pos2 = actor_state2.get("location", {})
        
        if not isinstance(pos1, dict) or not isinstance(pos2, dict):
            return 0.0
        
        try:
            x1, z1 = float(pos1.get("x", 0)), float(pos1.get("z", 0))
            x2, z2 = float(pos2.get("x", 0)), float(pos2.get("z", 0))
            return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
        except (ValueError, TypeError):
            return 0.0
    
    def _get_tower_hp_info(self, frame_data: Dict) -> Dict[str, float]:
        """获取防御塔血量信息"""
        tower_hp = {'my': 1.0, 'enemy': 1.0}
        
        for npc in frame_data.get("npc_states", []):
            if 'TOWER' in npc.get('sub_type', ''):
                hp_ratio = float(npc.get('hp', 0)) / max(float(npc.get('max_hp', 1)), 1)
                if npc.get('camp') == self.main_camp:
                    tower_hp['my'] = hp_ratio
                else:
                    tower_hp['enemy'] = hp_ratio
        
        return tower_hp
    
    def _count_last_hits(self, frame_data: Dict) -> int:
        """计算补刀数量"""
        last_hits = 0
        for action in frame_data.get("frame_action", []):
            dead_action = action.get("dead_action", {})
            death = dead_action.get("death", {})
            killer = dead_action.get("killer", {})
            
            if ("SOLDIER" in str(death.get("type", "")) and 
                killer.get("player_id") == self.main_hero_id):
                last_hits += 1
        
        return last_hits
    
    def _detect_combo_execution(self, main_hero: Dict) -> float:
        """检测连招执行 - 简化实现"""
        # 基于技能使用频率和命中率
        skill_state = main_hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        
        recent_skill_uses = 0
        for slot in slots[:3]:  # 前3个技能
            if slot.get("succUsedInFrame", 0) > 0:
                recent_skill_uses += 1
        
        return min(recent_skill_uses / 3.0, 1.0)
    
    def _calculate_position_advantage(self, main_hero: Dict, enemy_hero: Dict, frame_data: Dict) -> float:
        """计算位置优势 - 简化实现"""
        # 基于与地图元素的相对位置
        return 0.0  # 简化处理
    
    def _detect_hero_attack_action(self, frame_data: Dict, main_hero: Dict) -> bool:
        """检测攻击英雄行为"""
        # 检查是否对英雄造成了伤害
        current_damage = float(main_hero.get("totalHurtToHero", 0))
        if self.state_history:
            prev_damage = self.state_history[-1].get('damage_dealt', current_damage)
            return current_damage > prev_damage
        return False
    
    def _detect_farming_action(self, frame_data: Dict) -> bool:
        """检测发育行为"""
        return self._count_last_hits(frame_data) > 0
    
    def _assess_tempo_control(self, main_hero: Dict, enemy_hero: Dict, frame_data: Dict) -> float:
        """评估节奏控制 - 简化实现"""
        return 0.0
    
    def _assess_farming_quality(self, main_hero: Dict) -> float:
        """评估发育质量"""
        # 基于金币获取效率
        if self.state_history:
            current_gold = float(main_hero.get("money", 0))
            prev_gold = self.state_history[-1].get('my_gold', current_gold)
            gold_gain = current_gold - prev_gold
            return min(gold_gain / 100.0, 1.0)
        return 0.0
    
    def _assess_fighting_quality(self, main_hero: Dict, enemy_hero: Dict) -> float:
        """评估对战质量"""
        # 基于伤害交换比
        if self.state_history:
            my_damage_dealt = float(main_hero.get("totalHurtToHero", 0))
            my_damage_received = float(main_hero.get("totalBeHurtByHero", 0))
            
            prev_state = self.state_history[-1]
            damage_dealt_gain = my_damage_dealt - prev_state.get('damage_dealt', my_damage_dealt)
            damage_received_gain = my_damage_received - prev_state.get('damage_received', my_damage_received)
            
            if damage_received_gain > 0:
                exchange_ratio = damage_dealt_gain / damage_received_gain
                return min(exchange_ratio, 2.0) / 2.0
        
        return 0.0
    
    def _assess_finishing_quality(self, main_hero: Dict, enemy_hero: Dict) -> float:
        """评估终结质量"""
        # 基于推塔进度和击杀
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        if enemy_hp_ratio < 0.3:  # 敌人残血，应该终结
            return 1.0 - enemy_hp_ratio
        return 0.0
    
    def _detect_inefficient_actions(self, main_hero: Dict, enemy_hero: Dict, frame_data: Dict) -> float:
        """检测无效行动"""
        inefficiency_score = 0.0
        
        distance = self._calculate_distance(main_hero, enemy_hero)
        
        # 检测技能使用效率
        skill_state = main_hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        
        for slot in slots:
            if slot.get("succUsedInFrame", 0) > 0:  # 使用了技能
                # 如果距离太远使用了近程技能
                if distance > 800 and slot.get("slot_type") in ["SKILL_1"]:  # 假设技能1是近程
                    inefficiency_score += 0.5
        
        return inefficiency_score
    
    def _detect_resource_waste(self, main_hero: Dict, frame_data: Dict) -> float:
        """检测资源浪费"""
        waste_score = 0.0
        
        # 检测满蓝不使用技能的情况
        values = main_hero.get("actor_state", {}).get("values", {})
        mp_ratio = float(values.get("ep", 0)) / max(float(values.get("max_ep", 1)), 1)
        
        if mp_ratio > 0.9:  # 蓝量很满
            # 如果有可用技能但没使用
            skill_state = main_hero.get("skill_state", {})
            slots = skill_state.get("slot_states", [])
            
            available_skills = sum(1 for slot in slots if slot.get("usable", False))
            used_skills = sum(1 for slot in slots if slot.get("succUsedInFrame", 0) > 0)
            
            if available_skills > 0 and used_skills == 0:
                waste_score += 0.3
        
        return waste_score
    
    def _detect_positioning_errors(self, main_hero: Dict, enemy_hero: Dict) -> float:
        """检测位置错误"""
        error_score = 0.0
        
        my_hp_ratio = self._get_hp_ratio(main_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        distance = self._calculate_distance(main_hero, enemy_hero)
        
        # 低血量时距离敌人太近
        if my_hp_ratio < 0.3 and distance < 400:
            error_score += (0.3 - my_hp_ratio) * 2.0
        
        # 血量优势时距离敌人太远
        if my_hp_ratio > enemy_hp_ratio + 0.3 and distance > 1200:
            error_score += (my_hp_ratio - enemy_hp_ratio - 0.3)
        
        return error_score
    
    def _update_reward_history(self, main_hero: Dict, enemy_hero: Dict, frame_data: Dict, frame_no: int):
        """更新奖励历史状态"""
        current_state = {
            'frame_no': frame_no,
            'my_hp_ratio': self._get_hp_ratio(main_hero),
            'enemy_hp_ratio': self._get_hp_ratio(enemy_hero),
            'my_gold': float(main_hero.get("money", 0)),
            'enemy_gold': float(enemy_hero.get("money", 0)),
            'my_exp': float(main_hero.get("exp", 0)),
            'distance': self._calculate_distance(main_hero, enemy_hero),
            'damage_dealt': float(main_hero.get("totalHurtToHero", 0)),
            'damage_received': float(main_hero.get("totalBeHurtByHero", 0)),
            'tower_hp': self._get_tower_hp_info(frame_data)
        }
        
        # 添加技能状态
        skill_state = main_hero.get("skill_state", {})
        slots = skill_state.get("slot_states", [])
        for i, slot in enumerate(slots):
            current_state[f'skill_{i}_hit'] = int(slot.get("hitHeroTimes", 0))
            current_state[f'skill_{i}_used'] = int(slot.get("usedTimes", 0))
        
        self.state_history.append(current_state)
    
    def get_reward_summary(self) -> Dict[str, float]:
        """获取奖励统计摘要"""
        if not self.reward_history:
            return {}
        
        summary = {}
        total_rewards = defaultdict(list)
        
        for reward_dict in self.reward_history:
            for reward_type, value in reward_dict.items():
                total_rewards[reward_type].append(value)
        
        for reward_type, values in total_rewards.items():
            summary[f'{reward_type}_avg'] = np.mean(values)
            summary[f'{reward_type}_sum'] = np.sum(values)
            summary[f'{reward_type}_count'] = len(values)
        
        return summary
