#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Enhanced by Deep RL Analysis

草丛战术奖励系统
基于深度强化学习分析，实现完整的草丛奖励函数

核心理念：
1. 不直接奖励"进草"，而是奖励利用草丛达成的"战术目的"
2. 信息不对称创造、伤害规避、心理压力施加
3. 视野博弈、战术欺诈和心理压迫的完整建模
4. 孙尚香特色连招：1技能进草 -> 强化普攻 -> 立刻回草
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict


class BushTacticalRewards:
    """草丛战术奖励系统 - 实现完整的草丛奖励函数"""
    
    def __init__(self, main_hero_id: int, main_camp: str):
        self.main_hero_id = main_hero_id
        self.main_camp = main_camp
        self.enemy_camp = "PLAYERCAMP_2" if main_camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        
        # 历史状态追踪
        self.state_history = deque(maxlen=20)
        self.ambush_history = deque(maxlen=100)
        
        # 奖励权重配置
        self.reward_weights = {
            # 1. 成功伏击奖励 (Successful Ambush Rewards)
            'first_strike_bonus_multiplier': 3.0,   # 先手优势伤害倍数
            'free_poke_bonus': 50.0,                # 无伤消耗奖励
            'perfect_ambush_bonus': 100.0,          # 完美伏击奖励
            'stealth_attack_bonus': 30.0,           # 隐身攻击奖励
            
            # 2. 信息控制与心理压迫奖励 (Info Control & Pressure)
            'vision_control_per_second': 5.0,       # 视野压制持续奖励 (每秒)
            'psychological_pressure': 20.0,         # 心理压迫奖励
            'information_asymmetry_bonus': 15.0,    # 信息不对称奖励
            'enemy_hesitation_reward': 25.0,        # 敌方犹豫奖励
            
            # 3. 战术性规避与重置奖励 (Tactical Evasion & Reset)
            'aggro_reset_bonus': 40.0,              # 仇恨重置奖励
            'skill_dodge_bonus': 80.0,              # 躲避关键技能奖励
            'tower_aggro_reset': 60.0,              # 防御塔仇恨重置奖励
            'strategic_retreat_bonus': 35.0,        # 战略撤退奖励
            
            # 4. 孙尚香特色连招奖励 (Sun Shangxiang Special Combo)
            'sunshangxiang_bush_combo': 120.0,      # 孙尚香草丛连招奖励
            'enhanced_auto_from_bush': 60.0,        # 草丛强化普攻奖励
            'roll_bush_escape': 45.0,               # 翻滚进草逃脱奖励
            'bush_kiting_mastery': 70.0,            # 草丛风筝精通奖励
            
            # 5. 高级草丛战术奖励 (Advanced Bush Tactics)
            'bush_control_dominance': 50.0,         # 草丛控制统治奖励
            'vision_game_mastery': 40.0,            # 视野博弈精通奖励
            'timing_perfection': 35.0,              # 时机完美奖励
            'bush_mind_games': 45.0,                # 草丛心理博弈奖励
            
            # 6. 惩罚机制 (Penalty Mechanisms)
            'blind_bush_entry': -20.0,              # 盲目进草惩罚
            'poor_bush_timing': -25.0,              # 草丛时机不当惩罚
            'wasted_stealth_opportunity': -30.0,    # 浪费隐身机会惩罚
            'exposed_positioning': -15.0,           # 暴露位置惩罚
        }
        
        # 战术状态追踪
        self.tactical_tracker = BushTacticalTracker()
        
        # 事件检测器
        self.event_detector = BushEventDetector(main_hero_id, main_camp)
        
        # 草丛定义 (与processor一致)
        self.bush_definitions = {
            'top_bush': {'center': (10000, 8000), 'radius': 800, 'type': 'offensive'},
            'bottom_bush': {'center': (5000, 8000), 'radius': 800, 'type': 'defensive'}
        }
        
    def calculate_bush_tactical_rewards(self, frame_data: Dict, main_hero: Dict, 
                                      enemy_hero: Dict, frame_no: int,
                                      bush_features: List[float] = None) -> Dict[str, float]:
        """计算草丛战术奖励"""
        if not main_hero or not enemy_hero:
            return {}
        
        rewards = {}
        
        # 提取当前草丛状态
        current_state = self._extract_current_bush_state(frame_data, main_hero, enemy_hero)
        
        # 1. 成功伏击奖励
        ambush_rewards = self._calculate_ambush_rewards(frame_data, main_hero, enemy_hero, current_state)
        rewards.update(ambush_rewards)
        
        # 2. 信息控制与心理压迫奖励
        control_rewards = self._calculate_control_pressure_rewards(current_state, frame_no)
        rewards.update(control_rewards)
        
        # 3. 战术性规避与重置奖励
        evasion_rewards = self._calculate_evasion_reset_rewards(frame_data, main_hero, current_state)
        rewards.update(evasion_rewards)
        
        # 4. 孙尚香特色连招奖励
        combo_rewards = self._calculate_sunshangxiang_combo_rewards(frame_data, main_hero, current_state)
        rewards.update(combo_rewards)
        
        # 5. 高级草丛战术奖励
        advanced_rewards = self._calculate_advanced_tactics_rewards(current_state, frame_no)
        rewards.update(advanced_rewards)
        
        # 6. 惩罚机制
        penalty_rewards = self._calculate_bush_penalties(frame_data, main_hero, current_state)
        rewards.update(penalty_rewards)
        
        # 更新历史状态
        self._update_tactical_history(current_state, frame_no)
        
        return rewards
    
    def _calculate_ambush_rewards(self, frame_data: Dict, main_hero: Dict, 
                                enemy_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算成功伏击奖励"""
        rewards = {}
        
        # 检测先手优势攻击
        first_strike_damage = self.event_detector.detect_first_strike_from_stealth(
            frame_data, main_hero, current_state
        )
        
        if first_strike_damage > 0:
            # 先手优势伤害倍数奖励
            base_damage_reward = first_strike_damage * 0.1  # 基础伤害奖励
            first_strike_bonus = base_damage_reward * (self.reward_weights['first_strike_bonus_multiplier'] - 1)
            rewards['first_strike_bonus'] = first_strike_bonus
            
            # 完美伏击检测 (从隐身状态发起攻击且造成大量伤害)
            if first_strike_damage > 200:  # 高伤害阈值
                rewards['perfect_ambush_bonus'] = self.reward_weights['perfect_ambush_bonus']
        
        # 检测无伤消耗
        free_poke_success = self.event_detector.detect_free_poke(frame_data, main_hero, current_state)
        if free_poke_success:
            rewards['free_poke_bonus'] = self.reward_weights['free_poke_bonus']
        
        # 隐身攻击奖励 (任何从隐身状态发起的攻击)
        if current_state.get('attacked_from_stealth', False):
            rewards['stealth_attack_bonus'] = self.reward_weights['stealth_attack_bonus']
        
        return rewards
    
    def _calculate_control_pressure_rewards(self, current_state: Dict, frame_no: int) -> Dict[str, float]:
        """计算信息控制与心理压迫奖励"""
        rewards = {}
        
        # 视野压制持续奖励 (我在暗，敌在明)
        if current_state.get('vision_advantage', 0) > 0:
            # 每帧给予小奖励，累积成持续奖励
            rewards['vision_control_continuous'] = self.reward_weights['vision_control_per_second'] / 30.0
        
        # 信息不对称奖励 (敌方不知道我的位置)
        if not current_state.get('self_visible_to_enemy', True):
            enemy_uncertainty = current_state.get('enemy_uncertainty', 0.0)
            if enemy_uncertainty > 0.5:  # 敌方高度不确定我的位置
                asymmetry_bonus = enemy_uncertainty * self.reward_weights['information_asymmetry_bonus']
                rewards['information_asymmetry_bonus'] = asymmetry_bonus
        
        # 心理压迫奖励 (基于威慑效果)
        intimidation_strength = current_state.get('intimidation_strength', 0.0)
        if intimidation_strength > 0.3:
            pressure_reward = intimidation_strength * self.reward_weights['psychological_pressure']
            rewards['psychological_pressure'] = pressure_reward
        
        # 敌方犹豫奖励 (检测敌方行为异常，如不敢上前补刀)
        enemy_hesitation = self._detect_enemy_hesitation(current_state)
        if enemy_hesitation > 0:
            rewards['enemy_hesitation_reward'] = enemy_hesitation * self.reward_weights['enemy_hesitation_reward']
        
        return rewards
    
    def _calculate_evasion_reset_rewards(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算战术性规避与重置奖励"""
        rewards = {}
        
        # 仇恨重置奖励
        aggro_reset_value = self.event_detector.detect_aggro_reset(frame_data, main_hero, current_state)
        if aggro_reset_value > 0:
            reset_reward = aggro_reset_value * self.reward_weights['aggro_reset_bonus']
            rewards['aggro_reset_bonus'] = reset_reward
        
        # 躲避关键技能奖励
        skill_dodge_success = self.event_detector.detect_skill_dodge(frame_data, main_hero, current_state)
        if skill_dodge_success:
            rewards['skill_dodge_bonus'] = self.reward_weights['skill_dodge_bonus']
        
        # 防御塔仇恨重置
        tower_reset_success = self.event_detector.detect_tower_aggro_reset(frame_data, main_hero, current_state)
        if tower_reset_success:
            rewards['tower_aggro_reset'] = self.reward_weights['tower_aggro_reset']
        
        # 战略撤退奖励 (低血量时成功进草脱离危险)
        strategic_retreat = self._detect_strategic_retreat(main_hero, current_state)
        if strategic_retreat:
            rewards['strategic_retreat_bonus'] = self.reward_weights['strategic_retreat_bonus']
        
        return rewards
    
    def _calculate_sunshangxiang_combo_rewards(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算孙尚香特色连招奖励"""
        rewards = {}
        
        # 检测孙尚香草丛连招 (1技能进草 -> 强化普攻 -> 立刻回草)
        bush_combo_success = self.event_detector.detect_sunshangxiang_bush_combo(
            frame_data, main_hero, current_state
        )
        if bush_combo_success:
            rewards['sunshangxiang_bush_combo'] = self.reward_weights['sunshangxiang_bush_combo']
        
        # 草丛强化普攻奖励
        enhanced_auto_from_bush = self.event_detector.detect_enhanced_auto_from_bush(
            frame_data, main_hero, current_state
        )
        if enhanced_auto_from_bush:
            rewards['enhanced_auto_from_bush'] = self.reward_weights['enhanced_auto_from_bush']
        
        # 翻滚进草逃脱奖励
        roll_escape_success = self.event_detector.detect_roll_bush_escape(frame_data, main_hero, current_state)
        if roll_escape_success:
            rewards['roll_bush_escape'] = self.reward_weights['roll_bush_escape']
        
        # 草丛风筝精通奖励
        kiting_mastery = self._calculate_bush_kiting_mastery(current_state)
        if kiting_mastery > 0:
            rewards['bush_kiting_mastery'] = kiting_mastery * self.reward_weights['bush_kiting_mastery']
        
        return rewards
    
    def _calculate_advanced_tactics_rewards(self, current_state: Dict, frame_no: int) -> Dict[str, float]:
        """计算高级草丛战术奖励"""
        rewards = {}
        
        # 草丛控制统治奖励
        control_dominance = current_state.get('bush_control_advantage', 0.0)
        if control_dominance > 0.6:
            rewards['bush_control_dominance'] = control_dominance * self.reward_weights['bush_control_dominance']
        
        # 视野博弈精通奖励
        vision_game_skill = self._calculate_vision_game_mastery(current_state)
        if vision_game_skill > 0:
            rewards['vision_game_mastery'] = vision_game_skill * self.reward_weights['vision_game_mastery']
        
        # 时机完美奖励
        timing_perfection = current_state.get('tactical_timing_readiness', 0.0)
        if timing_perfection > 0.8:
            rewards['timing_perfection'] = timing_perfection * self.reward_weights['timing_perfection']
        
        # 草丛心理博弈奖励
        mind_game_success = self._detect_bush_mind_games(current_state)
        if mind_game_success > 0:
            rewards['bush_mind_games'] = mind_game_success * self.reward_weights['bush_mind_games']
        
        return rewards
    
    def _calculate_bush_penalties(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> Dict[str, float]:
        """计算草丛惩罚机制"""
        rewards = {}
        
        # 盲目进草惩罚 (在不合适的时机进入草丛)
        blind_entry_penalty = self._detect_blind_bush_entry(main_hero, current_state)
        if blind_entry_penalty > 0:
            rewards['blind_bush_entry'] = -blind_entry_penalty * abs(self.reward_weights['blind_bush_entry'])
        
        # 草丛时机不当惩罚
        poor_timing_penalty = self._detect_poor_bush_timing(current_state)
        if poor_timing_penalty > 0:
            rewards['poor_bush_timing'] = -poor_timing_penalty * abs(self.reward_weights['poor_bush_timing'])
        
        # 浪费隐身机会惩罚
        wasted_opportunity = self._detect_wasted_stealth_opportunity(current_state)
        if wasted_opportunity > 0:
            rewards['wasted_stealth_opportunity'] = -wasted_opportunity * abs(self.reward_weights['wasted_stealth_opportunity'])
        
        # 暴露位置惩罚 (不必要的暴露)
        exposed_positioning = self._detect_exposed_positioning(main_hero, current_state)
        if exposed_positioning > 0:
            rewards['exposed_positioning'] = -exposed_positioning * abs(self.reward_weights['exposed_positioning'])
        
        return rewards
    
    # ============ 核心检测方法 ============
    
    def _detect_enemy_hesitation(self, current_state: Dict) -> float:
        """检测敌方犹豫行为"""
        # 基于敌方位置变化的异常检测
        hesitation_score = 0.0
        
        # 如果我方在草丛中且敌方可见
        if not current_state.get('self_visible_to_enemy', True) and current_state.get('enemy_visible_to_self', False):
            # 检查敌方是否在补刀距离内但没有补刀
            enemy_near_minions = current_state.get('enemy_near_minions', False)
            enemy_last_hit_opportunity = current_state.get('enemy_last_hit_opportunity', 0)
            
            if enemy_near_minions and enemy_last_hit_opportunity > 0:
                # 有补刀机会但可能因为威慑而犹豫
                hesitation_score = 0.7
        
        return hesitation_score
    
    def _detect_strategic_retreat(self, main_hero: Dict, current_state: Dict) -> bool:
        """检测战略撤退"""
        my_hp_ratio = self._get_hp_ratio(main_hero)
        
        # 低血量且成功进入草丛
        if my_hp_ratio < 0.4 and current_state.get('entered_bush_this_frame', False):
            # 且之前处于危险状态 (被攻击或在危险位置)
            if current_state.get('was_in_danger', False):
                return True
        
        return False
    
    def _calculate_bush_kiting_mastery(self, current_state: Dict) -> float:
        """计算草丛风筝精通度"""
        mastery = 0.0
        
        # 检查是否在草丛附近进行风筝
        if current_state.get('near_bush', False):
            # 有效的进出草丛操作
            bush_transitions = current_state.get('bush_transitions', 0)
            if bush_transitions > 0:
                mastery += 0.3
            
            # 保持攻击输出的同时规避伤害
            damage_dealt = current_state.get('damage_dealt_this_frame', 0)
            damage_taken = current_state.get('damage_taken_this_frame', 0)
            
            if damage_dealt > 0 and damage_taken == 0:
                mastery += 0.5
            
            # 维持最佳攻击距离
            optimal_distance = current_state.get('optimal_distance_maintained', False)
            if optimal_distance:
                mastery += 0.2
        
        return min(mastery, 1.0)
    
    def _calculate_vision_game_mastery(self, current_state: Dict) -> float:
        """计算视野博弈精通度"""
        mastery = 0.0
        
        # 视野优势的维持
        vision_advantage = current_state.get('vision_advantage', 0.0)
        if vision_advantage > 0:
            mastery += vision_advantage * 0.4
        
        # 视野信息的有效利用
        info_utilization = current_state.get('information_utilization', 0.0)
        mastery += info_utilization * 0.3
        
        # 视野博弈的主动权
        vision_initiative = current_state.get('vision_initiative', 0.0)
        mastery += vision_initiative * 0.3
        
        return min(mastery, 1.0)
    
    def _detect_bush_mind_games(self, current_state: Dict) -> float:
        """检测草丛心理博弈成功"""
        mind_game_score = 0.0
        
        # 成功的视野欺诈
        if current_state.get('vision_deception_success', False):
            mind_game_score += 0.4
        
        # 迫使敌方做出错误决策
        if current_state.get('enemy_forced_mistake', False):
            mind_game_score += 0.6
        
        return mind_game_score
    
    def _detect_blind_bush_entry(self, main_hero: Dict, current_state: Dict) -> float:
        """检测盲目进草惩罚"""
        penalty = 0.0
        
        if current_state.get('entered_bush_this_frame', False):
            # 在不合适的时机进草 (如敌方有视野、技能全交等)
            if current_state.get('enemy_has_vision', False):
                penalty += 0.5
            
            # 没有明确的战术目的
            if not current_state.get('has_tactical_purpose', False):
                penalty += 0.3
            
            # 处于技能冷却中
            if not current_state.get('skills_available', False):
                penalty += 0.2
        
        return penalty
    
    def _detect_poor_bush_timing(self, current_state: Dict) -> float:
        """检测草丛时机不当"""
        penalty = 0.0
        
        # 在错误的时机使用草丛
        timing_readiness = current_state.get('tactical_timing_readiness', 1.0)
        if timing_readiness < 0.3:
            penalty = 1.0 - timing_readiness
        
        return penalty
    
    def _detect_wasted_stealth_opportunity(self, current_state: Dict) -> float:
        """检测浪费隐身机会"""
        penalty = 0.0
        
        # 在隐身状态下没有有效行动
        if not current_state.get('self_visible_to_enemy', True):
            # 有攻击机会但没有攻击
            if current_state.get('attack_opportunity', 0.0) > 0.7:
                if not current_state.get('attacked_this_frame', False):
                    penalty = current_state['attack_opportunity']
        
        return penalty
    
    def _detect_exposed_positioning(self, main_hero: Dict, current_state: Dict) -> float:
        """检测暴露位置惩罚"""
        penalty = 0.0
        
        # 在不必要的情况下暴露位置
        if current_state.get('self_visible_to_enemy', False):
            # 附近有草丛可以躲避但没有使用
            if current_state.get('bush_available_nearby', False):
                # 且当前处于不利状态
                if current_state.get('in_disadvantageous_state', False):
                    penalty = 0.8
        
        return penalty
    
    # ============ 辅助方法 ============
    
    def _extract_current_bush_state(self, frame_data: Dict, main_hero: Dict, enemy_hero: Dict) -> Dict:
        """提取当前草丛状态"""
        # 这里需要与bush_tactical_processor的特征提取保持一致
        state = {
            'my_pos': self._get_position(main_hero),
            'enemy_pos': self._get_position(enemy_hero),
            'self_visible_to_enemy': True,  # 简化实现，实际需要复杂计算
            'enemy_visible_to_self': True,
            'entered_bush_this_frame': False,
            'vision_advantage': 0.0,
            'intimidation_strength': 0.0,
            'bush_control_advantage': 0.0,
            'tactical_timing_readiness': 0.5,
            'attack_opportunity': 0.0,
            'ambush_opportunity': 0.0,
        }
        
        return state
    
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
    
    def _update_tactical_history(self, current_state: Dict, frame_no: int):
        """更新战术历史"""
        history_entry = current_state.copy()
        history_entry['frame_no'] = frame_no
        
        self.state_history.append(history_entry)


class BushTacticalTracker:
    """草丛战术状态追踪器"""
    
    def __init__(self):
        self.bush_activity_log = deque(maxlen=300)
        self.ambush_success_count = 0
        self.vision_control_duration = 0
        
    def update_activity(self, state: Dict, frame_no: int):
        """更新草丛活动"""
        activity = {
            'frame_no': frame_no,
            'activity_type': self._classify_activity(state),
            'success': self._evaluate_success(state)
        }
        self.bush_activity_log.append(activity)
    
    def _classify_activity(self, state: Dict) -> str:
        """分类草丛活动"""
        if state.get('ambush_opportunity', 0) > 0.7:
            return 'ambush'
        elif state.get('vision_advantage', 0) > 0:
            return 'vision_control'
        elif state.get('entered_bush_this_frame', False):
            return 'bush_entry'
        else:
            return 'observation'
    
    def _evaluate_success(self, state: Dict) -> bool:
        """评估活动成功性"""
        # 简化实现
        return state.get('tactical_success', False)


class BushEventDetector:
    """草丛事件检测器"""
    
    def __init__(self, main_hero_id: int, main_camp: str):
        self.main_hero_id = main_hero_id
        self.main_camp = main_camp
        self.recent_events = deque(maxlen=50)
        
    def detect_first_strike_from_stealth(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> float:
        """检测从隐身状态发起的先手攻击"""
        # 检查是否从隐身状态攻击敌方英雄
        if not current_state.get('self_visible_to_enemy', True):
            damage = self._get_hero_damage_dealt(frame_data, main_hero)
            if damage > 0:
                return damage
        return 0.0
    
    def detect_free_poke(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> bool:
        """检测无伤消耗成功"""
        # 检查是否攻击了敌方但没有受到反击
        damage_dealt = self._get_hero_damage_dealt(frame_data, main_hero)
        damage_taken = self._get_hero_damage_received(frame_data, main_hero)
        
        # 从隐身状态攻击且没有受到伤害
        if damage_dealt > 0 and damage_taken == 0 and not current_state.get('self_visible_to_enemy', True):
            return True
        
        return False
    
    def detect_aggro_reset(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> float:
        """检测仇恨重置成功"""
        # 简化实现：检查是否通过进草丛重置了小兵仇恨
        if current_state.get('entered_bush_this_frame', False):
            # 检查之前是否被多个小兵攻击
            previous_minion_aggro = current_state.get('previous_minion_aggro', 0)
            if previous_minion_aggro >= 2:
                return min(previous_minion_aggro / 5.0, 1.0)
        
        return 0.0
    
    def detect_skill_dodge(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> bool:
        """检测技能躲避成功"""
        # 简化实现：检查是否通过进草躲避了大额伤害
        if current_state.get('entered_bush_this_frame', False):
            # 检查是否避免了预期的高伤害
            expected_damage = current_state.get('expected_skill_damage', 0)
            actual_damage = self._get_hero_damage_received(frame_data, main_hero)
            
            if expected_damage > 150 and actual_damage < expected_damage * 0.3:
                return True
        
        return False
    
    def detect_tower_aggro_reset(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> bool:
        """检测防御塔仇恨重置"""
        # 简化实现
        return False
    
    def detect_sunshangxiang_bush_combo(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> bool:
        """检测孙尚香草丛连招"""
        # 检查是否执行了1技能进草 -> 强化普攻的连招
        # 这需要基于技能使用和位置变化的复杂逻辑
        # 简化实现
        return False
    
    def detect_enhanced_auto_from_bush(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> bool:
        """检测草丛强化普攻"""
        # 简化实现
        return False
    
    def detect_roll_bush_escape(self, frame_data: Dict, main_hero: Dict, current_state: Dict) -> bool:
        """检测翻滚进草逃脱"""
        # 简化实现
        return False
    
    def _get_hero_damage_dealt(self, frame_data: Dict, hero: Dict) -> float:
        """获取英雄造成的伤害"""
        hero_id = hero.get("player_id", -1)
        total_damage = 0.0
        
        for action in frame_data.get("frame_action", []):
            hurt_action = action.get("hurt_action", {})
            if hurt_action:
                attacker_id = hurt_action.get("attacker_player_id", -1)
                damage = float(hurt_action.get("damage", 0))
                
                if attacker_id == hero_id:
                    total_damage += damage
        
        return total_damage
    
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
