#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
兵线宏观特征处理模块 (Wave Process)
专门处理小兵兵线的宏观态势、推进深度、补刀机会等高层次特征

Author: Enhanced AI Arena Features
"""

import math
import configparser
import os
from typing import Dict, Any, List, Tuple
from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer


class WaveProcess:
    def __init__(self, camp: str):
        """
        camp: 'PLAYERCAMP_1' or 'PLAYERCAMP_2'
        """
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.mirror = (camp == "PLAYERCAMP_2")
        
        # 配置参数
        self.SOLDIER_TYPE = "ACTOR_SUB_SOLDIER"
        self.CANNON_CONFIG_IDS = [101, 102, 103]  # 假设的炮车配置ID
        self.MELEE_CONFIG_IDS = [201, 202, 203]   # 假设的近战兵配置ID
        self.RANGED_CONFIG_IDS = [301, 302, 303]  # 假设的远程兵配置ID
        
        # 地图常量
        self.MAP_SIZE = 60000.0
        self.LANE_LENGTH = 40000.0
        
        # 加载配置
        self.get_wave_config()

    def get_wave_config(self):
        """加载兵线特征配置"""
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "wave_feature_config.ini")
        
        # 如果配置文件不存在，创建默认配置
        if not os.path.exists(config_path):
            self._create_default_config(config_path)
        
        self.config.read(config_path)
        
        # 获取归一化配置
        self.wave_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.wave_feature_config.append(f"{feature}:{config}")
        
        self.map_feature_to_norm = self.normalizer.parse_config(self.wave_feature_config)

    def _create_default_config(self, config_path: str):
        """创建默认的兵线特征配置文件"""
        config_content = """[feature_config]
# 兵线宏观状态特征 (12维)
wave_allied_total_hp = min_max:0:20000
wave_enemy_total_hp = min_max:0:20000
wave_hp_advantage = min_max:-20000:20000
wave_allied_count = min_max:0:15
wave_enemy_count = min_max:0:15
wave_count_advantage = min_max:-15:15
wave_allied_cannon_exist = one_hot:1:eq
wave_enemy_cannon_exist = one_hot:1:eq
wave_allied_dps = min_max:0:5000
wave_enemy_dps = min_max:0:5000
wave_value_advantage = min_max:-10000:10000
wave_push_power = min_max:-1:1

# 兵线空间与位置特征 (10维)
wave_frontline_dist_to_enemy_tower = min_max:0:40000
wave_frontline_dist_to_my_tower = min_max:0:40000
wave_clash_point_x = min_max:-30000:30000
wave_clash_point_z = min_max:-30000:30000
hero_dist_to_clash_point = min_max:0:40000
hero_dist_to_nearest_enemy_minion = min_max:0:40000
wave_spread_factor = min_max:0:1
wave_lane_control = min_max:-1:1
wave_retreat_distance = min_max:0:10000
minion_formation_quality = min_max:0:1

# 兵线目标与攻击意图特征 (8维)
wave_enemy_aggro_hero_count = min_max:0:15
wave_allied_aggro_hero_count = min_max:0:15
is_tower_aggro_cannon = one_hot:1:eq
wave_dps_on_enemy_hero = min_max:0:3000
wave_dps_on_allied_hero = min_max:0:3000
tower_focus_priority = min_max:0:1
minion_aggro_efficiency = min_max:0:1
wave_threat_level = min_max:0:1

# 补刀与收益预测特征 (12维)
wave_enemy_last_hittable_count = min_max:0:15
wave_lowest_hp_enemy_minion_hp = min_max:0:2000
wave_is_tower_about_to_last_hit = one_hot:1:eq
wave_gold_potential = min_max:0:1000
wave_exp_potential = min_max:0:500
next_wave_arrival_time = min_max:0:10000
wave_last_hit_competition = min_max:0:1
cannon_last_hit_timing = min_max:0:5000
denied_gold_potential = min_max:0:500
farming_efficiency_score = min_max:0:1
wave_rhythm_control = min_max:0:1
optimal_clear_timing = min_max:0:1

# 兵线战术与时机特征 (8维)
wave_dive_safety = min_max:0:1
wave_freeze_opportunity = min_max:0:1
wave_slow_push_value = min_max:0:1
wave_fast_push_value = min_max:0:1
wave_bounce_prediction = min_max:0:1
minion_block_advantage = min_max:0:1
wave_zone_control = min_max:0:1
wave_transition_timing = min_max:0:1

[feature_functions]
# 这里配置特征函数映射，实际在代码中实现
"""
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

    def process_wave_features(self, frame_state: Dict[str, Any], hero_info: Dict[str, Any]) -> List[float]:
        """
        处理兵线宏观特征
        返回50维特征向量
        """
        # 获取小兵列表
        ally_soldiers, enemy_soldiers = self._get_soldier_lists(frame_state)
        
        # 获取防御塔信息
        ally_tower, enemy_tower = self._get_tower_info(frame_state)
        
        # 计算各类特征
        wave_features = []
        
        # 1. 兵线宏观状态特征 (12维)
        wave_features.extend(self._calculate_wave_macro_features(ally_soldiers, enemy_soldiers))
        
        # 2. 兵线空间与位置特征 (10维)
        wave_features.extend(self._calculate_spatial_features(ally_soldiers, enemy_soldiers, ally_tower, enemy_tower, hero_info))
        
        # 3. 兵线目标与攻击意图特征 (8维)
        wave_features.extend(self._calculate_targeting_features(ally_soldiers, enemy_soldiers, hero_info, frame_state))
        
        # 4. 补刀与收益预测特征 (12维)
        wave_features.extend(self._calculate_last_hit_features(ally_soldiers, enemy_soldiers, hero_info, ally_tower, enemy_tower))
        
        # 5. 兵线战术与时机特征 (8维)
        wave_features.extend(self._calculate_tactical_features(ally_soldiers, enemy_soldiers, hero_info, frame_state))
        
        return wave_features

    def _get_soldier_lists(self, frame_state: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """获取己方和敌方小兵列表"""
        ally_soldiers = []
        enemy_soldiers = []
        
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == self.SOLDIER_TYPE and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        return ally_soldiers, enemy_soldiers

    def _get_tower_info(self, frame_state: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """获取防御塔信息"""
        ally_tower = None
        enemy_tower = None
        
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_TOWER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_tower = npc
                else:
                    enemy_tower = npc
        
        return ally_tower, enemy_tower

    def _calculate_wave_macro_features(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> List[float]:
        """计算兵线宏观状态特征 (12维)"""
        features = []
        
        # 基础血量和数量统计
        ally_total_hp = sum(s.get("hp", 0) for s in ally_soldiers)
        enemy_total_hp = sum(s.get("hp", 0) for s in enemy_soldiers)
        ally_count = len(ally_soldiers)
        enemy_count = len(enemy_soldiers)
        
        features.extend([
            ally_total_hp,                    # wave_allied_total_hp
            enemy_total_hp,                   # wave_enemy_total_hp
            ally_total_hp - enemy_total_hp,   # wave_hp_advantage
            ally_count,                       # wave_allied_count
            enemy_count,                      # wave_enemy_count
            ally_count - enemy_count,         # wave_count_advantage
        ])
        
        # 炮车存在性
        ally_cannon_exist = 1.0 if any(self._is_cannon(s) for s in ally_soldiers) else 0.0
        enemy_cannon_exist = 1.0 if any(self._is_cannon(s) for s in enemy_soldiers) else 0.0
        features.extend([ally_cannon_exist, enemy_cannon_exist])
        
        # DPS计算
        ally_dps = sum(s.get("phy_atk", 0) + s.get("mgc_atk", 0) for s in ally_soldiers)
        enemy_dps = sum(s.get("phy_atk", 0) + s.get("mgc_atk", 0) for s in enemy_soldiers)
        features.extend([ally_dps, enemy_dps])
        
        # 兵线价值优势（基于小兵类型和血量）
        ally_value = self._calculate_wave_value(ally_soldiers)
        enemy_value = self._calculate_wave_value(enemy_soldiers)
        value_advantage = ally_value - enemy_value
        features.append(value_advantage)
        
        # 推进力量（综合评估）
        push_power = self._calculate_push_power(ally_soldiers, enemy_soldiers)
        features.append(push_power)
        
        return features

    def _calculate_spatial_features(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict], 
                                  ally_tower: Dict, enemy_tower: Dict, hero_info: Dict) -> List[float]:
        """计算兵线空间与位置特征 (10维)"""
        features = []
        
        # 前线距离计算
        if ally_soldiers and enemy_tower:
            frontline_dist_to_enemy_tower = min(
                self._distance(s.get("location", {}), enemy_tower.get("location", {})) 
                for s in ally_soldiers
            )
        else:
            frontline_dist_to_enemy_tower = 40000.0
        
        if enemy_soldiers and ally_tower:
            frontline_dist_to_my_tower = min(
                self._distance(s.get("location", {}), ally_tower.get("location", {}))
                for s in enemy_soldiers
            )
        else:
            frontline_dist_to_my_tower = 40000.0
        
        features.extend([frontline_dist_to_enemy_tower, frontline_dist_to_my_tower])
        
        # 兵线交战点
        clash_point = self._calculate_clash_point(ally_soldiers, enemy_soldiers)
        if self.mirror:
            clash_point = (-clash_point[0], -clash_point[1])
        features.extend(clash_point)
        
        # 英雄到兵线距离
        if hero_info:
            hero_pos = hero_info.get("actor_state", {}).get("location", {})
            hero_dist_to_clash = self._distance(hero_pos, {"x": clash_point[0], "z": clash_point[1]})
            
            if enemy_soldiers:
                hero_dist_to_nearest_enemy = min(
                    self._distance(hero_pos, s.get("location", {}))
                    for s in enemy_soldiers
                )
            else:
                hero_dist_to_nearest_enemy = 40000.0
        else:
            hero_dist_to_clash = 40000.0
            hero_dist_to_nearest_enemy = 40000.0
        
        features.extend([hero_dist_to_clash, hero_dist_to_nearest_enemy])
        
        # 兵线分散程度
        spread_factor = self._calculate_spread_factor(ally_soldiers + enemy_soldiers)
        features.append(spread_factor)
        
        # 兵线控制权
        lane_control = self._calculate_lane_control(ally_soldiers, enemy_soldiers, ally_tower, enemy_tower)
        features.append(lane_control)
        
        # 撤退距离安全性
        retreat_distance = self._calculate_retreat_distance(ally_soldiers, ally_tower)
        features.append(retreat_distance)
        
        # 小兵阵型质量
        formation_quality = self._calculate_formation_quality(ally_soldiers)
        features.append(formation_quality)
        
        return features

    def _calculate_targeting_features(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict], 
                                    hero_info: Dict, frame_state: Dict) -> List[float]:
        """计算兵线目标与攻击意图特征 (8维)"""
        features = []
        
        # 获取英雄runtime_id
        hero_runtime_id = None
        enemy_hero_runtime_id = None
        if hero_info:
            hero_runtime_id = hero_info.get("actor_state", {}).get("runtime_id")
        
        # 找敌方英雄
        for hero in frame_state.get("hero_states", []):
            if hero.get("actor_state", {}).get("camp") != self.main_camp:
                enemy_hero_runtime_id = hero.get("actor_state", {}).get("runtime_id")
                break
        
        # 小兵攻击英雄统计
        enemy_aggro_hero_count = 0
        allied_aggro_hero_count = 0
        
        for soldier in enemy_soldiers:
            if soldier.get("attack_target") == hero_runtime_id:
                enemy_aggro_hero_count += 1
        
        for soldier in ally_soldiers:
            if soldier.get("attack_target") == enemy_hero_runtime_id:
                allied_aggro_hero_count += 1
        
        features.extend([enemy_aggro_hero_count, allied_aggro_hero_count])
        
        # 塔攻击炮车检查
        is_tower_aggro_cannon = 0.0
        ally_cannon_ids = [s.get("runtime_id") for s in ally_soldiers if self._is_cannon(s)]
        for npc in frame_state.get("npc_states", []):
            if (npc.get("sub_type") == "ACTOR_SUB_TOWER" and 
                npc.get("camp") != self.main_camp and
                npc.get("attack_target") in ally_cannon_ids):
                is_tower_aggro_cannon = 1.0
                break
        
        features.append(is_tower_aggro_cannon)
        
        # DPS计算
        dps_on_enemy_hero = sum(
            s.get("phy_atk", 0) + s.get("mgc_atk", 0) 
            for s in ally_soldiers 
            if s.get("attack_target") == enemy_hero_runtime_id
        )
        
        dps_on_allied_hero = sum(
            s.get("phy_atk", 0) + s.get("mgc_atk", 0)
            for s in enemy_soldiers
            if s.get("attack_target") == hero_runtime_id
        )
        
        features.extend([dps_on_enemy_hero, dps_on_allied_hero])
        
        # 塔攻击优先级
        tower_focus_priority = self._calculate_tower_focus_priority(ally_soldiers, frame_state)
        features.append(tower_focus_priority)
        
        # 小兵仇恨效率
        aggro_efficiency = self._calculate_aggro_efficiency(ally_soldiers, enemy_soldiers)
        features.append(aggro_efficiency)
        
        # 兵线威胁等级
        threat_level = self._calculate_threat_level(enemy_soldiers, hero_info)
        features.append(threat_level)
        
        return features

    def _calculate_last_hit_features(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict],
                                   hero_info: Dict, ally_tower: Dict, enemy_tower: Dict) -> List[float]:
        """计算补刀与收益预测特征 (12维)"""
        features = []
        
        hero_atk = 0
        if hero_info:
            hero_atk = hero_info.get("actor_state", {}).get("values", {}).get("phy_atk", 0)
        
        # 可补刀小兵数量
        last_hittable_count = sum(1 for s in enemy_soldiers if s.get("hp", 0) <= hero_atk and s.get("hp", 0) > 0)
        features.append(last_hittable_count)
        
        # 最低血量敌方小兵
        if enemy_soldiers:
            lowest_hp = min(s.get("hp", 0) for s in enemy_soldiers if s.get("hp", 0) > 0)
        else:
            lowest_hp = 0
        features.append(lowest_hp)
        
        # 塔即将补刀检查
        is_tower_about_to_last_hit = self._check_tower_last_hit(ally_soldiers, enemy_tower)
        features.append(is_tower_about_to_last_hit)
        
        # 金币潜力
        gold_potential = sum(self._get_minion_gold_value(s) for s in enemy_soldiers)
        features.append(gold_potential)
        
        # 经验潜力
        exp_potential = sum(self._get_minion_exp_value(s) for s in enemy_soldiers)
        features.append(exp_potential)
        
        # 下一波兵到达时间（估算）
        next_wave_time = self._estimate_next_wave_time(ally_soldiers, enemy_soldiers)
        features.append(next_wave_time)
        
        # 补刀竞争激烈程度
        last_hit_competition = self._calculate_last_hit_competition(enemy_soldiers, hero_atk)
        features.append(last_hit_competition)
        
        # 炮车补刀时机
        cannon_timing = self._calculate_cannon_last_hit_timing(enemy_soldiers, hero_atk)
        features.append(cannon_timing)
        
        # 反补潜力
        denied_gold = sum(self._get_minion_gold_value(s) for s in ally_soldiers if s.get("hp", 0) <= hero_atk)
        features.append(denied_gold)
        
        # 刷兵效率评分
        farming_efficiency = self._calculate_farming_efficiency(enemy_soldiers, hero_info)
        features.append(farming_efficiency)
        
        # 兵线节奏控制
        rhythm_control = self._calculate_rhythm_control(ally_soldiers, enemy_soldiers)
        features.append(rhythm_control)
        
        # 最优清兵时机
        optimal_clear_timing = self._calculate_optimal_clear_timing(enemy_soldiers, hero_info)
        features.append(optimal_clear_timing)
        
        return features

    def _calculate_tactical_features(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict],
                                   hero_info: Dict, frame_state: Dict) -> List[float]:
        """计算兵线战术与时机特征 (8维)"""
        features = []
        
        # 越塔安全性
        dive_safety = self._calculate_dive_safety(ally_soldiers, enemy_soldiers, hero_info)
        features.append(dive_safety)
        
        # 控线机会
        freeze_opportunity = self._calculate_freeze_opportunity(ally_soldiers, enemy_soldiers)
        features.append(freeze_opportunity)
        
        # 慢推价值
        slow_push_value = self._calculate_slow_push_value(ally_soldiers, enemy_soldiers)
        features.append(slow_push_value)
        
        # 快推价值
        fast_push_value = self._calculate_fast_push_value(ally_soldiers, enemy_soldiers)
        features.append(fast_push_value)
        
        # 兵线回弹预测
        bounce_prediction = self._calculate_bounce_prediction(ally_soldiers, enemy_soldiers)
        features.append(bounce_prediction)
        
        # 小兵卡位优势
        block_advantage = self._calculate_minion_block_advantage(ally_soldiers, hero_info)
        features.append(block_advantage)
        
        # 兵线区域控制
        zone_control = self._calculate_wave_zone_control(ally_soldiers, enemy_soldiers)
        features.append(zone_control)
        
        # 兵线转换时机
        transition_timing = self._calculate_transition_timing(ally_soldiers, enemy_soldiers, frame_state)
        features.append(transition_timing)
        
        return features

    # ======================= 辅助函数 ======================= #
    
    def _is_cannon(self, soldier: Dict) -> bool:
        """判断是否为炮车"""
        config_id = soldier.get("config_id", 0)
        return config_id in self.CANNON_CONFIG_IDS
    
    def _is_melee(self, soldier: Dict) -> bool:
        """判断是否为近战兵"""
        config_id = soldier.get("config_id", 0)
        return config_id in self.MELEE_CONFIG_IDS
    
    def _is_ranged(self, soldier: Dict) -> bool:
        """判断是否为远程兵"""
        config_id = soldier.get("config_id", 0)
        return config_id in self.RANGED_CONFIG_IDS

    def _distance(self, pos1: Dict, pos2: Dict) -> float:
        """计算两点距离"""
        x1, z1 = pos1.get("x", 0), pos1.get("z", 0)
        x2, z2 = pos2.get("x", 0), pos2.get("z", 0)
        return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)

    def _calculate_wave_value(self, soldiers: List[Dict]) -> float:
        """计算兵线价值（基于小兵类型和血量）"""
        total_value = 0.0
        for soldier in soldiers:
            base_value = soldier.get("max_hp", 0) * 0.1
            if self._is_cannon(soldier):
                base_value *= 3.0  # 炮车价值更高
            elif self._is_ranged(soldier):
                base_value *= 1.5  # 远程兵价值中等
            # 近战兵保持基础价值
            
            hp_ratio = soldier.get("hp", 0) / max(1, soldier.get("max_hp", 1))
            total_value += base_value * hp_ratio
        
        return total_value

    def _calculate_push_power(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """计算推进力量"""
        ally_power = len(ally_soldiers) + (2 if any(self._is_cannon(s) for s in ally_soldiers) else 0)
        enemy_power = len(enemy_soldiers) + (2 if any(self._is_cannon(s) for s in enemy_soldiers) else 0)
        
        if ally_power + enemy_power == 0:
            return 0.0
        
        return (ally_power - enemy_power) / (ally_power + enemy_power)

    def _calculate_clash_point(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> Tuple[float, float]:
        """计算兵线交战点"""
        if not ally_soldiers and not enemy_soldiers:
            return (0.0, 0.0)
        
        all_positions = []
        for soldiers in [ally_soldiers, enemy_soldiers]:
            for soldier in soldiers:
                pos = soldier.get("location", {})
                all_positions.append((pos.get("x", 0), pos.get("z", 0)))
        
        if not all_positions:
            return (0.0, 0.0)
        
        avg_x = sum(pos[0] for pos in all_positions) / len(all_positions)
        avg_z = sum(pos[1] for pos in all_positions) / len(all_positions)
        
        return (avg_x, avg_z)

    def _calculate_spread_factor(self, all_soldiers: List[Dict]) -> float:
        """计算兵线分散程度"""
        if len(all_soldiers) < 2:
            return 0.0
        
        positions = [(s.get("location", {}).get("x", 0), s.get("location", {}).get("z", 0)) for s in all_soldiers]
        
        # 计算位置方差
        avg_x = sum(pos[0] for pos in positions) / len(positions)
        avg_z = sum(pos[1] for pos in positions) / len(positions)
        
        variance = sum((pos[0] - avg_x) ** 2 + (pos[1] - avg_z) ** 2 for pos in positions) / len(positions)
        spread = math.sqrt(variance)
        
        return min(spread / 5000.0, 1.0)  # 归一化

    def _calculate_lane_control(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict],
                              ally_tower: Dict, enemy_tower: Dict) -> float:
        """计算兵线控制权"""
        if not ally_tower or not enemy_tower:
            return 0.0
        
        ally_center = ally_tower.get("location", {})
        enemy_center = enemy_tower.get("location", {})
        
        ally_avg_dist = 0.0
        enemy_avg_dist = 0.0
        
        if ally_soldiers:
            ally_avg_dist = sum(self._distance(s.get("location", {}), enemy_center) for s in ally_soldiers) / len(ally_soldiers)
        if enemy_soldiers:
            enemy_avg_dist = sum(self._distance(s.get("location", {}), ally_center) for s in enemy_soldiers) / len(enemy_soldiers)
        
        if ally_avg_dist + enemy_avg_dist == 0:
            return 0.0
        
        return (enemy_avg_dist - ally_avg_dist) / (ally_avg_dist + enemy_avg_dist)

    def _calculate_retreat_distance(self, ally_soldiers: List[Dict], ally_tower: Dict) -> float:
        """计算撤退距离安全性"""
        if not ally_soldiers or not ally_tower:
            return 0.0
        
        tower_pos = ally_tower.get("location", {})
        min_dist = min(self._distance(s.get("location", {}), tower_pos) for s in ally_soldiers)
        
        return min(min_dist / 10000.0, 1.0)

    def _calculate_formation_quality(self, soldiers: List[Dict]) -> float:
        """计算小兵阵型质量"""
        if len(soldiers) < 2:
            return 0.0
        
        # 简化实现：检查小兵是否聚集在合理范围内
        positions = [s.get("location", {}) for s in soldiers]
        center_x = sum(p.get("x", 0) for p in positions) / len(positions)
        center_z = sum(p.get("z", 0) for p in positions) / len(positions)
        
        max_dist = max(self._distance(p, {"x": center_x, "z": center_z}) for p in positions)
        
        # 理想的阵型应该在合理范围内
        ideal_range = 3000.0
        quality = max(0.0, 1.0 - max_dist / ideal_range)
        
        return quality

    def _calculate_tower_focus_priority(self, ally_soldiers: List[Dict], frame_state: Dict) -> float:
        """计算塔攻击优先级合理性"""
        # 简化实现
        return 0.5

    def _calculate_aggro_efficiency(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """计算小兵仇恨效率"""
        # 简化实现：基于攻击目标的合理性
        return 0.5

    def _calculate_threat_level(self, enemy_soldiers: List[Dict], hero_info: Dict) -> float:
        """计算兵线威胁等级"""
        if not enemy_soldiers or not hero_info:
            return 0.0
        
        hero_pos = hero_info.get("actor_state", {}).get("location", {})
        total_threat = 0.0
        
        for soldier in enemy_soldiers:
            soldier_pos = soldier.get("location", {})
            distance = self._distance(hero_pos, soldier_pos)
            
            # 距离越近威胁越大
            if distance < 3000:
                threat = (3000 - distance) / 3000.0
                total_threat += threat
        
        return min(total_threat / len(enemy_soldiers), 1.0)

    def _check_tower_last_hit(self, ally_soldiers: List[Dict], enemy_tower: Dict) -> float:
        """检查塔是否即将补刀"""
        if not ally_soldiers or not enemy_tower:
            return 0.0
        
        tower_atk = enemy_tower.get("phy_atk", 0)
        tower_target = enemy_tower.get("attack_target")
        
        for soldier in ally_soldiers:
            if (soldier.get("runtime_id") == tower_target and 
                soldier.get("hp", 0) <= tower_atk):
                return 1.0
        
        return 0.0

    def _get_minion_gold_value(self, soldier: Dict) -> float:
        """获取小兵金币价值"""
        if self._is_cannon(soldier):
            return 60.0
        elif self._is_ranged(soldier):
            return 20.0
        else:
            return 15.0  # 近战兵

    def _get_minion_exp_value(self, soldier: Dict) -> float:
        """获取小兵经验价值"""
        if self._is_cannon(soldier):
            return 40.0
        elif self._is_ranged(soldier):
            return 15.0
        else:
            return 10.0

    def _estimate_next_wave_time(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """估算下一波兵到达时间"""
        # 简化实现：基于当前兵线数量估算
        total_soldiers = len(ally_soldiers) + len(enemy_soldiers)
        if total_soldiers < 3:
            return 1000.0  # 兵线即将刷新
        else:
            return 5000.0  # 还有一段时间

    def _calculate_last_hit_competition(self, enemy_soldiers: List[Dict], hero_atk: float) -> float:
        """计算补刀竞争激烈程度"""
        if not enemy_soldiers:
            return 0.0
        
        low_hp_count = sum(1 for s in enemy_soldiers if s.get("hp", 0) < hero_atk * 2)
        return min(low_hp_count / len(enemy_soldiers), 1.0)

    def _calculate_cannon_last_hit_timing(self, enemy_soldiers: List[Dict], hero_atk: float) -> float:
        """计算炮车补刀时机"""
        for soldier in enemy_soldiers:
            if self._is_cannon(soldier):
                hp = soldier.get("hp", 0)
                if hp <= hero_atk * 3:  # 炮车血量较厚，需要提前准备
                    return (hero_atk * 3 - hp) / (hero_atk * 3)
        return 0.0

    def _calculate_farming_efficiency(self, enemy_soldiers: List[Dict], hero_info: Dict) -> float:
        """计算刷兵效率评分"""
        if not enemy_soldiers or not hero_info:
            return 0.0
        
        # 基于英雄攻击力和小兵血量分布评估效率
        hero_atk = hero_info.get("actor_state", {}).get("values", {}).get("phy_atk", 0)
        if hero_atk == 0:
            return 0.0
        
        efficient_targets = sum(1 for s in enemy_soldiers if hero_atk * 0.5 <= s.get("hp", 0) <= hero_atk * 2)
        return min(efficient_targets / len(enemy_soldiers), 1.0)

    def _calculate_rhythm_control(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """计算兵线节奏控制"""
        # 基于双方兵线数量和血量的平衡度
        ally_hp = sum(s.get("hp", 0) for s in ally_soldiers)
        enemy_hp = sum(s.get("hp", 0) for s in enemy_soldiers)
        
        total_hp = ally_hp + enemy_hp
        if total_hp == 0:
            return 0.5
        
        balance = abs(ally_hp - enemy_hp) / total_hp
        return 1.0 - balance  # 越平衡节奏控制越好

    def _calculate_optimal_clear_timing(self, enemy_soldiers: List[Dict], hero_info: Dict) -> float:
        """计算最优清兵时机"""
        if not enemy_soldiers or not hero_info:
            return 0.0
        
        # 简化实现：基于小兵血量分布判断是否适合清兵
        hero_atk = hero_info.get("actor_state", {}).get("values", {}).get("phy_atk", 0)
        low_hp_soldiers = sum(1 for s in enemy_soldiers if s.get("hp", 0) <= hero_atk * 1.5)
        
        return min(low_hp_soldiers / len(enemy_soldiers), 1.0)

    def _calculate_dive_safety(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict], hero_info: Dict) -> float:
        """计算越塔安全性"""
        if not ally_soldiers:
            return 0.0
        
        # 基于己方小兵数量和血量评估越塔安全性
        ally_count = len(ally_soldiers)
        enemy_count = len(enemy_soldiers)
        
        if ally_count >= enemy_count + 2:  # 兵线优势明显
            return 0.8
        elif ally_count > enemy_count:
            return 0.5
        else:
            return 0.2

    def _calculate_freeze_opportunity(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """计算控线机会"""
        # 控线需要敌方兵线稍多但不太多
        ally_count = len(ally_soldiers)
        enemy_count = len(enemy_soldiers)
        
        if 1 <= enemy_count - ally_count <= 2:
            return 1.0
        elif enemy_count > ally_count:
            return 0.5
        else:
            return 0.0

    def _calculate_slow_push_value(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """计算慢推价值"""
        # 慢推适合在己方小兵稍多的情况下
        ally_count = len(ally_soldiers)
        enemy_count = len(enemy_soldiers)
        
        if 1 <= ally_count - enemy_count <= 3:
            return 1.0
        else:
            return 0.0

    def _calculate_fast_push_value(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """计算快推价值"""
        # 快推适合在己方兵线优势明显时
        ally_count = len(ally_soldiers)
        enemy_count = len(enemy_soldiers)
        
        if ally_count >= enemy_count + 3:
            return 1.0
        elif ally_count > enemy_count + 1:
            return 0.6
        else:
            return 0.0

    def _calculate_bounce_prediction(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """计算兵线回弹预测"""
        # 简化实现：基于兵线推进深度预测回弹
        ally_count = len(ally_soldiers)
        enemy_count = len(enemy_soldiers)
        
        if enemy_count > ally_count + 2:
            return 0.8  # 敌方兵线强势，可能回弹
        else:
            return 0.2

    def _calculate_minion_block_advantage(self, ally_soldiers: List[Dict], hero_info: Dict) -> float:
        """计算小兵卡位优势"""
        # 简化实现：基于小兵位置和英雄位置的配合
        if not ally_soldiers or not hero_info:
            return 0.0
        
        # 这里可以实现更复杂的卡位判断逻辑
        return 0.5

    def _calculate_wave_zone_control(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict]) -> float:
        """计算兵线区域控制"""
        ally_count = len(ally_soldiers)
        enemy_count = len(enemy_soldiers)
        total_count = ally_count + enemy_count
        
        if total_count == 0:
            return 0.5
        
        return ally_count / total_count

    def _calculate_transition_timing(self, ally_soldiers: List[Dict], enemy_soldiers: List[Dict], frame_state: Dict) -> float:
        """计算兵线转换时机"""
        # 基于帧数和兵线状态判断是否适合转换战术
        frame_no = frame_state.get("frameNo", 0)
        
        # 简化实现：每隔一段时间提高转换机会
        cycle = frame_no % 3000  # 假设每100秒一个周期
        if cycle < 500:  # 周期开始时适合转换
            return 1.0
        else:
            return 0.3
