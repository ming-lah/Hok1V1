#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from enum import Enum
from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math
from collections import OrderedDict


class OrganProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp

        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}

        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_organ_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.organ_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 13  # 原来7维 + 新增6维塔特征 = 13维
        self.unit_buff_num = 1

        # 其他特征属性
        self.MAP_NORM = 30000.0
        self.RANGE_NORM = 15000.0
        self.MINION_TOPK = 6
        self.BULLET_TOPK = 4

    def get_organ_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "organ_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.organ_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.organ_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_organ(self, frame_state):
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        local_vector_feature = []


        # 1) 生成己方&敌方阵营的防御塔特征
        ally_camp_organ_vector_feature  = self.generate_one_type_organ_feature(self.main_camp_organ_dict, "ally_camp")
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.enemy_camp_organ_dict, "enemy_camp")
        local_vector_feature.extend(ally_camp_organ_vector_feature)
        local_vector_feature.extend(enemy_camp_organ_vector_feature)

        # 2) 小兵特征
        local_vector_feature.extend(self._encode_minions(frame_state))

        # 3) 子弹压力
        local_vector_feature.extend(self._encode_bullets(frame_state))

        # 4) BUff摘要
        local_vector_feature.extend(self._encode_buff_summary(frame_state))

        # 5) 事件/草丛
        local_vector_feature.extend(self._encode_events_and_grass(frame_state))

        # 6) 经济优势特征
        local_vector_feature.extend(self._encode_economic_advantage(frame_state))

        # 7) 游戏阶段特征  
        local_vector_feature.extend(self._encode_game_phase_features(frame_state))

        # 8) 兵线宏观特征 - 新增的核心特征
        local_vector_feature.extend(self._encode_wave_macro_features(frame_state))

        # 9) 兵线空间特征
        local_vector_feature.extend(self._encode_wave_spatial_features(frame_state))

        # 10) 兵线目标特征
        local_vector_feature.extend(self._encode_wave_targeting_features(frame_state))

        # 11) 兵线补刀特征
        local_vector_feature.extend(self._encode_wave_lasthit_features(frame_state))

        # 12) 防御塔高级特征 - 新增的核心塔特征
        local_vector_feature.extend(self._encode_tower_intrinsic_features(frame_state))

        # 13) 防御塔英雄交互特征
        local_vector_feature.extend(self._encode_tower_hero_interaction_features(frame_state))

        # 14) 防御塔兵线交互特征
        local_vector_feature.extend(self._encode_tower_minion_interaction_features(frame_state))

        # 15) 防御塔预测特征
        local_vector_feature.extend(self._encode_tower_predictive_features(frame_state))

        # 16) 野怪特征 - 全新的战略资源控制特征
        local_vector_feature.extend(self._encode_jungle_monster_features(frame_state))

        # 17) 高级防御塔特征 - 动态风险评估与战略机会分析
        local_vector_feature.extend(self._encode_advanced_tower_features(frame_state))

        # 18) 专家级兵线特征 - 兵线健康度分析与高级控制预测
        local_vector_feature.extend(self._encode_expert_wave_features(frame_state))

        vector_feature = local_vector_feature
        return vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["actor_state"]["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["actor_state"]["config_id"]] = hero

    def generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()

        for organ in frame_state["npc_states"]:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == self.main_camp:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    self.main_camp_organ_dict["tower"] = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    self.enemy_camp_organ_dict["tower"] = organ

    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        num_organs_considered = 0

        def process_organ(organ):
            nonlocal num_organs_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](organ, value)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_organs_considered += 1

        if "tower" in one_type_organ_info:
            organ = one_type_organ_info["tower"]
            process_organ(organ)

        if num_organs_considered < self.unit_buff_num:
            self.no_organ_feature(vector_feature, num_organs_considered)
        return vector_feature

    def no_organ_feature(self, vector_feature, num_organs_considered):
        for _ in range((self.unit_buff_num - num_organs_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def get_hp_rate(self, organ, vector_feature):
        value = 0
        if organ["max_hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    # =================== ENHANCED TOWER FEATURES =================== #
    def tower_aggro_status(self, organ, vector_feature):
        """防御塔仇恨状态"""
        # 检查塔是否正在攻击英雄
        aggro_value = 0.0
        if organ and organ.get("hp", 0) > 0:
            # 简化实现：检查塔的攻击目标是否为英雄
            attack_target = organ.get("attack_target", -1)
            # 如果攻击目标存在且不是小兵，可能是英雄
            if attack_target > 0:
                aggro_value = 1.0
        vector_feature.append(aggro_value)

    def tower_damage_threat(self, organ, vector_feature):
        """塔伤害威胁等级"""
        threat_value = 0.0
        if organ and organ.get("hp", 0) > 0:
            # 基于塔的攻击力和己方英雄的位置计算威胁
            tower_attack = organ.get("phy_atk", 0)  # 塔的物理攻击力
            if hasattr(self, 'main_hero_info') and self.main_hero_info:
                hero_pos = self.main_hero_info["actor_state"]["location"]
                tower_pos = organ["location"]
                distance = self.cal_dist(hero_pos, tower_pos)
                
                # 塔的攻击范围通常是固定的
                tower_range = organ.get("attack_range", 8000)  # 默认8000
                
                if distance <= tower_range:
                    # 在塔攻击范围内，威胁等级基于攻击力
                    threat_value = min(tower_attack / 2000.0, 1.0)  # 归一化
                else:
                    threat_value = 0.0
        vector_feature.append(threat_value)

    def tower_dive_opportunity(self, organ, vector_feature):
        """越塔击杀机会"""
        dive_score = 0.0
        if organ and organ.get("hp", 0) > 0:
            # 评估越塔的可行性
            if hasattr(self, 'main_hero_info') and self.main_hero_info:
                # 己方英雄血量
                hero_hp = self.main_hero_info["actor_state"]["hp"]
                hero_max_hp = self.main_hero_info["actor_state"]["max_hp"]
                hero_hp_ratio = hero_hp / max(1, hero_max_hp)
                
                # 敌方英雄血量（从敌方英雄字典获取）
                enemy_hp_ratio = 1.0
                for enemy in self.enemy_camp_hero_dict.values():
                    enemy_hp = enemy["actor_state"]["hp"]
                    enemy_max_hp = enemy["actor_state"]["max_hp"]
                    enemy_hp_ratio = enemy_hp / max(1, enemy_max_hp)
                    break
                
                # 越塔机会：己方血量充足 + 敌方血量低
                if hero_hp_ratio > 0.6 and enemy_hp_ratio < 0.3:
                    dive_score = (hero_hp_ratio - 0.6) * 2.5 + (0.3 - enemy_hp_ratio) * 3.33
                    dive_score = min(dive_score, 1.0)
        vector_feature.append(dive_score)

    def tower_protection_value(self, organ, vector_feature):
        """塔保护价值"""
        protection_value = 0.0
        if organ and organ.get("hp", 0) > 0:
            # 己方塔的保护价值基于血量和战略位置
            if organ["camp"] == self.main_camp:
                hp_ratio = organ["hp"] / max(1, organ["max_hp"])
                protection_value = hp_ratio  # 血量越高保护价值越大
            else:
                # 敌方塔的保护价值（对敌方而言）
                hp_ratio = organ["hp"] / max(1, organ["max_hp"])
                protection_value = 1.0 - hp_ratio  # 敌塔血量越低，我方推塔价值越大
        vector_feature.append(protection_value)

    def tower_push_timing(self, organ, vector_feature):
        """推塔时机评估"""
        push_timing = 0.0
        if organ and organ.get("hp", 0) > 0 and organ["camp"] != self.main_camp:
            # 评估推敌方塔的时机
            # 基于小兵波次和敌方英雄状态
            ally_minions_nearby = 0
            enemy_hero_nearby = False
            
            if hasattr(self, 'main_hero_info') and self.main_hero_info:
                hero_pos = self.main_hero_info["actor_state"]["location"]
                tower_pos = organ["location"]
                
                # 检查附近小兵数量（简化）
                # 这里可以通过遍历frame_state中的小兵来计算
                ally_minions_nearby = 3  # 假设值，实际应该计算
                
                # 检查敌方英雄是否在附近
                for enemy in self.enemy_camp_hero_dict.values():
                    enemy_pos = enemy["actor_state"]["location"]
                    dist_to_tower = self.cal_dist(enemy_pos, tower_pos)
                    if dist_to_tower < 1000:  # 敌方英雄在塔附近
                        enemy_hero_nearby = True
                        break
                
                # 推塔时机：有小兵掩护 + 敌方英雄不在附近
                if ally_minions_nearby >= 2 and not enemy_hero_nearby:
                    push_timing = min(ally_minions_nearby / 5.0, 1.0)
        vector_feature.append(push_timing)

    def minion_tank_availability(self, organ, vector_feature):
        """小兵坦克可用性"""
        tank_value = 0.0
        if organ and organ.get("hp", 0) > 0:
            if hasattr(self, 'main_hero_info') and self.main_hero_info:
                # 简化实现：假设附近己方小兵可以作为肉盾
                # 实际实现应该检查frame_state中的小兵位置和血量
                hero_pos = self.main_hero_info["actor_state"]["location"]
                tower_pos = organ["location"]
                dist_to_tower = self.cal_dist(hero_pos, tower_pos)
                
                if dist_to_tower < 1500:  # 在塔附近
                    # 假设有小兵可用（实际应该计算）
                    tank_value = 0.7  # 简化值
        vector_feature.append(tank_value)

    def judge_in_view(self, main_hero_location, obj_location):
        if (
            (main_hero_location["x"] - obj_location["x"] >= 0 - self.view_dist)
            and (main_hero_location["x"] - obj_location["x"] <= self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] >= 0 - self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] <= self.view_dist)
        ):
            return True
        return False

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def is_alive(self, organ, vector_feature):
        value = 0.0
        if organ["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def belong_to_main_camp(self, organ, vector_feature):
        value = 0.0
        if organ["camp"] == self.main_hero_info["actor_state"]["camp"]:
            value = 1.0
        vector_feature.append(value)

    def get_normal_organ_location_x(self, organ, vector_feature):
        value = organ["location"]["x"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_normal_organ_location_z(self, organ, vector_feature):
        value = organ["location"]["z"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def relative_location_x(self, organ, vector_feature):
        organ_location_x = organ["location"]["x"]
        location_x = self.main_hero_info["actor_state"]["location"]["x"]
        x_diff = organ_location_x - location_x
        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def relative_location_z(self, organ, vector_feature):
        organ_location_z = organ["location"]["z"]
        location_z = self.main_hero_info["actor_state"]["location"]["z"]
        z_diff = organ_location_z - location_z
        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def _encode_minions(self, frame_state):
        """
        输出：5 + k*5 + k*5
        汇总(5)：[a_cnt_norm, e_cnt_norm, a_hp_sum_norm, e_hp_sum_norm, push_depth_norm]
        TopK单槽(5)：[alive, dx_norm, dz_norm, hp_ratio, atk_tower_flag]
        """
        npc = frame_state.get("npc_states", []) or []
        A = [u for u in npc if u.get("camp") == self.main_camp and u.get("sub_type") == "ACTOR_SUB_SOLDIER" and u.get("hp",0)>0]
        E = [u for u in npc if u.get("camp") != self.main_camp and u.get("sub_type") == "ACTOR_SUB_SOLDIER" and u.get("hp",0)>0]

        def _hp_sum(lst): return float(sum(max(0.0, float(u.get("hp",0.0))) for u in lst))
        a_cnt = min(len(A), 10) / 10.0
        e_cnt = min(len(E), 10) / 10.0
        a_hpsum = min(_hp_sum(A), 20000.0) / 20000.0
        e_hpsum = min(_hp_sum(E), 20000.0) / 20000.0

        def _min_dist_to(target, lst):
            if not lst or not target: return self.RANGE_NORM
            tx, tz = self._pos(target); res = self.RANGE_NORM
            for u in lst:
                ux, uz = self._pos(u)
                d = math.hypot(ux-tx, uz-tz)
                if d < res: res = d
            return min(res, self.RANGE_NORM)
        ally_tower  = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        a_front = _min_dist_to(enemy_tower, A)
        e_front = _min_dist_to(ally_tower, E)
        push_depth = (e_front - a_front) / self.RANGE_NORM  # [-1,1]
        me = self.main_hero_info["actor_state"]
        mx, mz = self._pos(me)
        def _dx_norm(x):
            if self.transform_camp2_to_camp1: x = -x
            return (x + 15000.0) / 30000.0
        def _slot(u, target_tower_id):
            if not u:
                return [0.0]*5
            ux, uz = self._pos(u)
            dx = ux - mx; dz = uz - mz
            hp = float(u.get("hp",0.0)); mxhp = float(u.get("max_hp",1.0))
            hp_ratio = 0.0 if mxhp <= 0 else (hp/mxhp)
            atk_tower_flag = 1.0 if int(u.get("attack_target", -1)) == int((target_tower_id or -2)) else 0.0
            return [1.0, _dx_norm(dx), _dx_norm(dz), hp_ratio, atk_tower_flag]
        def _topk(lst):
            return sorted(lst, key=lambda u: math.hypot(self._pos(u)[0]-mx, self._pos(u)[1]-mz))[:self.MINION_TOPK]

        ally_slots  = sum((_slot(u, (enemy_tower or {}).get("runtime_id")) for u in _topk(A)), [])
        enemy_slots = sum((_slot(u, (ally_tower  or {}).get("runtime_id")) for u in _topk(E)), [])
        while len(ally_slots)  < self.MINION_TOPK*5: ally_slots += [0.0]*5
        while len(enemy_slots) < self.MINION_TOPK*5: enemy_slots += [0.0]*5

        return [a_cnt, e_cnt, a_hpsum, e_hpsum, push_depth] + ally_slots + enemy_slots
    
    def _encode_bullets(self, frame_state):
        """
        输出：2 + K*4 维
        汇总(2)：[bullet_cnt_norm, incoming_cnt_norm]
        TopK来弹单槽(4)：[dx_norm, dz_norm, dist_norm, cos_to_me]
        """
        me = self.main_hero_info["actor_state"]
        mx, mz = self._pos(me)
        bullets = frame_state.get("bullets", []) or []
        enemy_bullets = [b for b in bullets if b.get("camp") == self.main_camp]

        def _cos_to_me(b):
            bx, bz = self._pos(b); v = b.get("use_dir") or {}
            vx, vz = float(v.get("x",0.0)), float(v.get("z",0.0))
            to_me_x, to_me_z = (mx - bx), (mz - bz)
            num = vx*to_me_x + vz*to_me_z
            den = math.hypot(vx, vz) * math.hypot(to_me_x, to_me_z)
            return 0.0 if den <= 1e-6 else max(-1.0, min(1.0, num/den))
        incoming = [b for b in enemy_bullets if _cos_to_me(b) > 0.0]

        cnt_norm = min(len(enemy_bullets), 20) / 20.0
        incoming_cnt_norm = min(len(incoming), 10) / 10.0

        def _dx_norm(x):
            if self.transform_camp2_to_camp1: x = -x
            return (x + 15000.0) / 30000.0
        def _score(b):
            bx, bz = self._pos(b)
            dist = math.hypot(bx-mx, bz-mz)
            return (-dist, _cos_to_me(b))
        incoming_sorted = sorted(incoming, key=_score)[:self.BULLET_TOPK]

        slots = []
        for b in incoming_sorted:
            bx, bz = self._pos(b)
            dx = _dx_norm(bx - mx); dz = _dx_norm(bz - mz)
            dist_norm = min(math.hypot(bx-mx, bz-mz), self.RANGE_NORM)/self.RANGE_NORM
            slots += [dx, dz, dist_norm, _cos_to_me(b)]
        while len(slots) < self.BULLET_TOPK*4: slots += [0.0]*4

        return [cnt_norm, incoming_cnt_norm] + slots
    
    def _encode_buff_summary(self, frame_state):
        """
        每英雄： [buff_mark_cnt_norm, buff_mark_layers_norm, buff_skill_group_cnt_norm]
        总计 6 维
        """
        # 敌方阵营（内联，不引入新函数）
        if self.main_camp == "PLAYERCAMP_1":
            enemy_camp = "PLAYERCAMP_2"
        elif self.main_camp == "PLAYERCAMP_2":
            enemy_camp = "PLAYERCAMP_1"
        elif isinstance(self.main_camp, int) and self.main_camp in (0, 1):
            enemy_camp = 1 - self.main_camp
        elif isinstance(self.main_camp, int) and self.main_camp in (1, 2):
            enemy_camp = 3 - self.main_camp
        else:
            enemy_camp = "PLAYERCAMP_2"

        def _hero(camp_value):
            for h in frame_state.get("hero_states", []) or []:
                if (h.get("actor_state") or {}).get("camp") == camp_value:
                    return h
            return None

        def _buff_vec(h):
            if not h:
                return [0.0, 0.0, 0.0]
            bs = h.get("buff_state") or (h.get("actor_state") or {}).get("buff_state") or {}
            marks = bs.get("buff_marks") or []
            skills = bs.get("buff_skills") or []
            cnt = min(len(marks), 10) / 10.0
            layer_sum = sum(int(m.get("layer", 0)) for m in marks)
            layer_norm = min(layer_sum, 10) / 10.0
            skill_cnt_norm = min(len(skills), 10) / 10.0
            return [cnt, layer_norm, skill_cnt_norm]

        ally = _buff_vec(_hero(self.main_camp))
        enemy = _buff_vec(_hero(enemy_camp))
        return ally + enemy


    def _encode_events_and_grass(self, frame_state):
        ally_dead = 0.0
        enemy_dead = 0.0

        # —— 事件列表兜底，保证 list[dict] —— 
        acts = frame_state.get("frame_action", []) or []
        if not isinstance(acts, list):
            acts = []
        acts = [a for a in acts if isinstance(a, dict)]

        # 敌方阵营（内联）
        if self.main_camp == "PLAYERCAMP_1":
            enemy_camp = "PLAYERCAMP_2"
        elif self.main_camp == "PLAYERCAMP_2":
            enemy_camp = "PLAYERCAMP_1"
        elif isinstance(self.main_camp, int) and self.main_camp in (0, 1):
            enemy_camp = 1 - self.main_camp
        elif isinstance(self.main_camp, int) and self.main_camp in (1, 2):
            enemy_camp = 3 - self.main_camp
        else:
            enemy_camp = "PLAYERCAMP_2"

        # 本帧死亡事件（直接等值比较）
        for act in acts:
            da = act.get("dead_action") or {}
            if not isinstance(da, dict):  # 再兜一下
                continue
            death = (da.get("death") or {}).get("camp", None)
            if death is None:
                continue
            if death == self.main_camp:
                ally_dead = 1.0
            elif death == enemy_camp:
                enemy_dead = 1.0

        # 英雄对象（己方优先用 generate_hero_info_list 生成的，敌方从 frame_state 找）
        ally_h = getattr(self, "main_hero_info", None)
        if not isinstance(ally_h, dict):
            ally_h = None

        enemy_h = None
        hs = frame_state.get("hero_states", []) or []
        if isinstance(hs, list):
            for h in hs:
                if isinstance(h, dict) and (h.get("actor_state") or {}).get("camp") == enemy_camp:
                    enemy_h = h
                    break

        grass_engage = 0.0
        if ally_h and enemy_h:
            me_in = 1 if (ally_h.get("isInGrass") or (ally_h.get("actor_state") or {}).get("isInGrass")) else 0
            en_in = 1 if (enemy_h.get("isInGrass") or (enemy_h.get("actor_state") or {}).get("isInGrass")) else 0
            if me_in and not en_in:
                ax, az = self._pos(ally_h.get("actor_state") or {})   # ← 修正变量名
                ex, ez = self._pos(enemy_h.get("actor_state") or {})
                if math.hypot(ax - ex, az - ez) <= 5000.0:
                    grass_engage = 1.0

        return [ally_dead, enemy_dead, grass_engage]

    # =================== ENHANCED ECONOMIC FEATURES =================== #
    def _encode_economic_advantage(self, frame_state):
        """
        经济优势特征 (6维)
        [gold_advantage, exp_advantage, item_power_diff, farming_efficiency, economic_trend, next_item_timing]
        """
        def _hero(camp_value):
            for h in frame_state.get("hero_states", []) or []:
                if (h.get("actor_state") or {}).get("camp") == camp_value:
                    return h
            return None

        # 敌方阵营
        if self.main_camp == "PLAYERCAMP_1":
            enemy_camp = "PLAYERCAMP_2"
        elif self.main_camp == "PLAYERCAMP_2":
            enemy_camp = "PLAYERCAMP_1"
        else:
            enemy_camp = "PLAYERCAMP_2"

        ally_h = _hero(self.main_camp)
        enemy_h = _hero(enemy_camp)

        def _safe_get_hero_value(h, key, default=0.0):
            if not h:
                return float(default)
            return float(h.get(key, (h.get("actor_state") or {}).get(key, default)))

        # 1. 金币优势
        my_gold = _safe_get_hero_value(ally_h, "money")
        enemy_gold = _safe_get_hero_value(enemy_h, "money")
        gold_advantage = (my_gold - enemy_gold) / max(5000.0, my_gold + enemy_gold)  # 归一化

        # 2. 经验优势
        my_exp = _safe_get_hero_value(ally_h, "exp")
        enemy_exp = _safe_get_hero_value(enemy_h, "exp")
        exp_advantage = (my_exp - enemy_exp) / max(1000.0, my_exp + enemy_exp)

        # 3. 装备战力差距（基于攻击力作为代理）
        my_atk = 0.0
        enemy_atk = 0.0
        if ally_h:
            my_phy_atk = float((ally_h.get("actor_state") or {}).get("values", {}).get("phy_atk", 0))
            my_mgc_atk = float((ally_h.get("actor_state") or {}).get("values", {}).get("mgc_atk", 0))
            my_atk = my_phy_atk + my_mgc_atk
        if enemy_h:
            enemy_phy_atk = float((enemy_h.get("actor_state") or {}).get("values", {}).get("phy_atk", 0))
            enemy_mgc_atk = float((enemy_h.get("actor_state") or {}).get("values", {}).get("mgc_atk", 0))
            enemy_atk = enemy_phy_atk + enemy_mgc_atk
        
        item_power_diff = (my_atk - enemy_atk) / max(1000.0, my_atk + enemy_atk)

        # 4. 发育效率（基于等级）
        my_level = float((ally_h or {}).get("level", 1))
        enemy_level = float((enemy_h or {}).get("level", 1))
        level_advantage = (my_level - enemy_level) / max(10.0, my_level + enemy_level)

        # 5. 经济趋势（简化：基于当前金币收入速率）
        my_kill_income = _safe_get_hero_value(ally_h, "kill_income")
        economic_trend = min(my_kill_income / 300.0, 1.0)  # 归一化

        # 6. 下一个大件时机
        next_item_cost = 2000.0  # 假设下一个大件2000金币
        next_item_timing = min(my_gold / next_item_cost, 1.0)

        return [
            max(-1.0, min(1.0, gold_advantage)),
            max(-1.0, min(1.0, exp_advantage)),
            max(-1.0, min(1.0, item_power_diff)),
            max(-1.0, min(1.0, level_advantage)),
            economic_trend,
            next_item_timing
        ]

    def _encode_game_phase_features(self, frame_state):
        """
        游戏阶段特征 (4维)
        [game_phase, tempo_advantage, initiative_control, late_game_scaling]
        """
        frame_no = frame_state.get("frameNo", 0)
        
        # 1. 游戏阶段（基于帧数）
        # 假设游戏30帧/秒，10分钟 = 18000帧
        if frame_no < 6000:      # 前3分钟：早期
            game_phase = 0.2
        elif frame_no < 12000:   # 3-6分钟：中期
            game_phase = 0.5
        elif frame_no < 18000:   # 6-10分钟：中后期
            game_phase = 0.8
        else:                    # 10分钟后：后期
            game_phase = 1.0

        # 2. 节奏优势（基于击杀数和推塔进度）
        def _hero(camp_value):
            for h in frame_state.get("hero_states", []) or []:
                if (h.get("actor_state") or {}).get("camp") == camp_value:
                    return h
            return None

        if self.main_camp == "PLAYERCAMP_1":
            enemy_camp = "PLAYERCAMP_2"
        else:
            enemy_camp = "PLAYERCAMP_1"

        ally_h = _hero(self.main_camp)
        enemy_h = _hero(enemy_camp)

        my_kills = float((ally_h or {}).get("killCnt", 0))
        enemy_kills = float((enemy_h or {}).get("killCnt", 0))
        kill_advantage = (my_kills - enemy_kills) / max(5.0, my_kills + enemy_kills + 1.0)

        # 塔血量优势
        ally_tower_hp = 1.0
        enemy_tower_hp = 1.0
        if hasattr(self, 'main_camp_organ_dict') and "tower" in self.main_camp_organ_dict:
            tower = self.main_camp_organ_dict["tower"]
            ally_tower_hp = tower["hp"] / max(1, tower["max_hp"])
        if hasattr(self, 'enemy_camp_organ_dict') and "tower" in self.enemy_camp_organ_dict:
            tower = self.enemy_camp_organ_dict["tower"]
            enemy_tower_hp = tower["hp"] / max(1, tower["max_hp"])

        tower_advantage = ally_tower_hp - enemy_tower_hp
        tempo_advantage = (kill_advantage * 0.6 + tower_advantage * 0.4)

        # 3. 主动权控制（基于位置和状态）
        initiative_score = 0.5  # 默认中性
        if ally_h and enemy_h:
            # 基于血量和位置判断主动权
            my_hp_ratio = float((ally_h.get("actor_state") or {}).get("hp", 0)) / max(1, float((ally_h.get("actor_state") or {}).get("max_hp", 1)))
            enemy_hp_ratio = float((enemy_h.get("actor_state") or {}).get("hp", 0)) / max(1, float((enemy_h.get("actor_state") or {}).get("max_hp", 1)))
            
            if my_hp_ratio > enemy_hp_ratio + 0.2:  # 血量优势
                initiative_score = 0.8
            elif enemy_hp_ratio > my_hp_ratio + 0.2:  # 血量劣势
                initiative_score = 0.2

        # 4. 后期成长性（基于英雄类型和装备，简化实现）
        late_game_scaling = 0.7  # 假设值，实际应基于英雄类型

        return [game_phase, max(-1.0, min(1.0, tempo_advantage)), initiative_score, late_game_scaling]

    # =================== ENHANCED WAVE/MINION FEATURES =================== #
    def _encode_wave_macro_features(self, frame_state):
        """
        兵线宏观特征 (8维) - 核心兵线状态
        """
        # 获取小兵列表
        ally_soldiers = []
        enemy_soldiers = []
        
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        # 基础统计
        wave_allied_total_hp = sum(s.get("hp", 0) for s in ally_soldiers)
        wave_enemy_total_hp = sum(s.get("hp", 0) for s in enemy_soldiers)
        wave_hp_advantage = wave_allied_total_hp - wave_enemy_total_hp
        
        wave_allied_count = len(ally_soldiers)
        wave_enemy_count = len(enemy_soldiers)
        wave_count_advantage = wave_allied_count - wave_enemy_count
        
        # 炮车检测（假设炮车config_id在101-105范围）
        def _is_cannon(soldier):
            config_id = soldier.get("config_id", 0)
            return config_id in [101, 102, 103, 104, 105]
        
        wave_allied_cannon_exist = 1.0 if any(_is_cannon(s) for s in ally_soldiers) else 0.0
        wave_enemy_cannon_exist = 1.0 if any(_is_cannon(s) for s in enemy_soldiers) else 0.0
        
        return [
            wave_allied_total_hp / 20000.0,  # 归一化
            wave_enemy_total_hp / 20000.0,
            max(-1.0, min(1.0, wave_hp_advantage / 20000.0)),
            wave_allied_count / 15.0,
            wave_enemy_count / 15.0,
            max(-1.0, min(1.0, wave_count_advantage / 15.0)),
            wave_allied_cannon_exist,
            wave_enemy_cannon_exist,
        ]

    def _encode_wave_spatial_features(self, frame_state):
        """
        兵线空间特征 (8维) - 位置与推进深度
        """
        # 获取小兵列表
        ally_soldiers = []
        enemy_soldiers = []
        
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        ally_tower = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        
        # 前线推进深度
        if ally_soldiers and enemy_tower:
            frontline_dist_to_enemy_tower = min(
                self.cal_dist(s.get("location", {}), enemy_tower.get("location", {}))
                for s in ally_soldiers
            )
        else:
            frontline_dist_to_enemy_tower = 30000.0
        
        if enemy_soldiers and ally_tower:
            frontline_dist_to_my_tower = min(
                self.cal_dist(s.get("location", {}), ally_tower.get("location", {}))
                for s in enemy_soldiers
            )
        else:
            frontline_dist_to_my_tower = 30000.0
        
        # 兵线交战点计算
        if ally_soldiers and enemy_soldiers:
            # 找到最前排的兵
            ally_frontline = min(ally_soldiers, key=lambda s: 
                self.cal_dist(s.get("location", {}), enemy_tower.get("location", {}) if enemy_tower else {"x": 0, "z": 0}))
            enemy_frontline = min(enemy_soldiers, key=lambda s: 
                self.cal_dist(s.get("location", {}), ally_tower.get("location", {}) if ally_tower else {"x": 0, "z": 0}))
            
            ally_pos = ally_frontline.get("location", {})
            enemy_pos = enemy_frontline.get("location", {})
            
            clash_point_x = (ally_pos.get("x", 0) + enemy_pos.get("x", 0)) / 2.0
            clash_point_z = (ally_pos.get("z", 0) + enemy_pos.get("z", 0)) / 2.0
            
            if self.transform_camp2_to_camp1:
                clash_point_x = -clash_point_x
                clash_point_z = -clash_point_z
        else:
            clash_point_x = 0.0
            clash_point_z = 0.0
        
        # 英雄到兵线距离
        if hasattr(self, 'main_hero_info') and self.main_hero_info:
            hero_pos = self.main_hero_info["actor_state"]["location"]
            hero_dist_to_clash_point = self.cal_dist(hero_pos, {"x": clash_point_x, "z": clash_point_z})
            
            if enemy_soldiers:
                hero_dist_to_nearest_enemy_minion = min(
                    self.cal_dist(hero_pos, s.get("location", {}))
                    for s in enemy_soldiers
                )
            else:
                hero_dist_to_nearest_enemy_minion = 30000.0
        else:
            hero_dist_to_clash_point = 30000.0
            hero_dist_to_nearest_enemy_minion = 30000.0
        
        return [
            frontline_dist_to_enemy_tower / 30000.0,   # 归一化到[0,1]
            frontline_dist_to_my_tower / 30000.0,
            clash_point_x / 30000.0,
            clash_point_z / 30000.0,
            hero_dist_to_clash_point / 30000.0,
            hero_dist_to_nearest_enemy_minion / 30000.0,
            0.5,  # wave_spread_factor (可后续完善)
            0.5,  # wave_lane_control (可后续完善)
        ]

    def _encode_wave_targeting_features(self, frame_state):
        """
        兵线目标与攻击意图特征 (8维) - 仇恨与DPS
        """
        # 获取小兵列表
        ally_soldiers = []
        enemy_soldiers = []
        
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        # 获取英雄runtime_id
        hero_runtime_id = None
        enemy_hero_runtime_id = None
        
        if hasattr(self, 'main_hero_info') and self.main_hero_info:
            hero_runtime_id = self.main_hero_info["actor_state"].get("runtime_id")
        
        # 找敌方英雄
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            enemy_hero_runtime_id = hero["actor_state"].get("runtime_id")
            break
        
        # 统计攻击英雄的小兵数量
        wave_enemy_aggro_hero_count = sum(
            1 for soldier in enemy_soldiers 
            if soldier.get("attack_target") == hero_runtime_id
        )
        
        wave_allied_aggro_hero_count = sum(
            1 for soldier in ally_soldiers 
            if soldier.get("attack_target") == enemy_hero_runtime_id
        )
        
        # 塔攻击炮车检查
        def _is_cannon(soldier):
            config_id = soldier.get("config_id", 0)
            return config_id in [101, 102, 103, 104, 105]
        
        is_tower_aggro_cannon = 0.0
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        if enemy_tower:
            tower_target = enemy_tower.get("attack_target")
            for soldier in ally_soldiers:
                if _is_cannon(soldier) and soldier.get("runtime_id") == tower_target:
                    is_tower_aggro_cannon = 1.0
                    break
        
        # DPS计算
        wave_dps_on_enemy_hero = sum(
            soldier.get("phy_atk", 0) + soldier.get("mgc_atk", 0)
            for soldier in ally_soldiers
            if soldier.get("attack_target") == enemy_hero_runtime_id
        )
        
        wave_dps_on_allied_hero = sum(
            soldier.get("phy_atk", 0) + soldier.get("mgc_atk", 0)
            for soldier in enemy_soldiers
            if soldier.get("attack_target") == hero_runtime_id
        )
        
        return [
            wave_enemy_aggro_hero_count / 15.0,  # 归一化
            wave_allied_aggro_hero_count / 15.0,
            is_tower_aggro_cannon,
            wave_dps_on_enemy_hero / 3000.0,
            wave_dps_on_allied_hero / 3000.0,
            0.5,  # tower_focus_priority (可后续完善)
            0.5,  # minion_aggro_efficiency (可后续完善)
            0.5,  # wave_threat_level (可后续完善)
        ]

    def _encode_wave_lasthit_features(self, frame_state):
        """
        补刀与收益预测特征 (8维) - 经济机会
        """
        # 获取小兵列表
        ally_soldiers = []
        enemy_soldiers = []
        
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        # 获取英雄攻击力
        hero_atk = 0
        if hasattr(self, 'main_hero_info') and self.main_hero_info:
            hero_values = self.main_hero_info["actor_state"].get("values", {})
            hero_atk = hero_values.get("phy_atk", 0)
        
        # 可补刀的敌方小兵数量
        wave_enemy_last_hittable_count = sum(
            1 for s in enemy_soldiers 
            if s.get("hp", 0) <= hero_atk and s.get("hp", 0) > 0
        )
        
        # 最低血量敌方小兵
        if enemy_soldiers:
            wave_lowest_hp_enemy_minion_hp = min(s.get("hp", 0) for s in enemy_soldiers if s.get("hp", 0) > 0)
        else:
            wave_lowest_hp_enemy_minion_hp = 0
        
        # 塔即将补刀检查
        wave_is_tower_about_to_last_hit = 0.0
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        if enemy_tower:
            tower_atk = enemy_tower.get("phy_atk", 0)
            tower_target = enemy_tower.get("attack_target")
            for soldier in ally_soldiers:
                if (soldier.get("runtime_id") == tower_target and 
                    soldier.get("hp", 0) <= tower_atk):
                    wave_is_tower_about_to_last_hit = 1.0
                    break
        
        # 金币和经验潜力
        def _get_minion_gold_value(soldier):
            config_id = soldier.get("config_id", 0)
            if config_id in [101, 102, 103, 104, 105]:  # 炮车
                return 60.0
            else:
                return 20.0  # 普通小兵
        
        wave_gold_potential = sum(_get_minion_gold_value(s) for s in enemy_soldiers)
        wave_exp_potential = sum(15.0 for s in enemy_soldiers)  # 简化的经验计算
        
        # 下一波兵到达时间（估算）
        total_soldiers = len(ally_soldiers) + len(enemy_soldiers)
        next_wave_arrival_time = 5000.0 if total_soldiers > 3 else 1000.0
        
        # 补刀竞争激烈程度
        if enemy_soldiers:
            low_hp_soldiers = sum(1 for s in enemy_soldiers if s.get("hp", 0) < hero_atk * 2)
            wave_last_hit_competition = low_hp_soldiers / len(enemy_soldiers)
        else:
            wave_last_hit_competition = 0.0
        
        return [
            wave_enemy_last_hittable_count / 15.0,  # 归一化
            wave_lowest_hp_enemy_minion_hp / 2000.0,
            wave_is_tower_about_to_last_hit,
            wave_gold_potential / 1000.0,
            wave_exp_potential / 500.0,
            next_wave_arrival_time / 10000.0,
            wave_last_hit_competition,
            0.5,  # 其他特征占位符
        ]

    # =================== ENHANCED TOWER ADVANCED FEATURES =================== #
    def _encode_tower_intrinsic_features(self, frame_state):
        """
        防御塔自身状态特征 (8维) - 核心塔状态
        """
        ally_tower = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        
        # 基础塔血量比例
        my_tower_hp_ratio = 0.0
        enemy_tower_hp_ratio = 0.0
        
        if ally_tower and ally_tower.get("max_hp", 0) > 0:
            my_tower_hp_ratio = ally_tower.get("hp", 0) / ally_tower["max_hp"]
        
        if enemy_tower and enemy_tower.get("max_hp", 0) > 0:
            enemy_tower_hp_ratio = enemy_tower.get("hp", 0) / enemy_tower["max_hp"]
        
        # 塔是否正在攻击
        my_tower_is_attacking = 0.0
        enemy_tower_is_attacking = 0.0
        
        # 通过npc_states检查攻击状态
        for npc in frame_state.get("npc_states", []):
            if ally_tower and npc.get("runtime_id") == ally_tower.get("runtime_id"):
                attack_target = npc.get("attack_target", -1)
                my_tower_is_attacking = 1.0 if attack_target and attack_target != -1 else 0.0
            elif enemy_tower and npc.get("runtime_id") == enemy_tower.get("runtime_id"):
                attack_target = npc.get("attack_target", -1)
                enemy_tower_is_attacking = 1.0 if attack_target and attack_target != -1 else 0.0
        
        # 塔攻击力和攻击速度
        my_tower_attack_power = ally_tower.get("phy_atk", 0) if ally_tower else 0
        enemy_tower_attack_power = enemy_tower.get("phy_atk", 0) if enemy_tower else 0
        my_tower_attack_speed = ally_tower.get("atk_spd", 1000) if ally_tower else 1000
        enemy_tower_attack_speed = enemy_tower.get("atk_spd", 1000) if enemy_tower else 1000
        
        return [
            my_tower_hp_ratio,
            enemy_tower_hp_ratio,
            my_tower_is_attacking,
            enemy_tower_is_attacking,
            my_tower_attack_power / 2000.0,  # 归一化
            enemy_tower_attack_power / 2000.0,
            my_tower_attack_speed / 3000.0,  # 归一化
            enemy_tower_attack_speed / 3000.0,
        ]

    def _encode_tower_hero_interaction_features(self, frame_state):
        """
        防御塔与英雄交互特征 (10维) - 英雄塔交互
        """
        ally_tower = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        
        # 获取英雄信息
        my_hero = None
        enemy_hero = None
        if hasattr(self, 'main_hero_info') and self.main_hero_info:
            my_hero = self.main_hero_info
        
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            enemy_hero = hero
            break
        
        # 英雄到塔的距离
        hero_dist_to_enemy_tower = 30000.0
        hero_dist_to_my_tower = 30000.0
        enemy_hero_dist_to_my_tower = 30000.0
        
        if my_hero and enemy_tower:
            hero_pos = my_hero["actor_state"]["location"]
            enemy_tower_pos = enemy_tower.get("location", {})
            hero_dist_to_enemy_tower = self.cal_dist(hero_pos, enemy_tower_pos)
        
        if my_hero and ally_tower:
            hero_pos = my_hero["actor_state"]["location"]
            ally_tower_pos = ally_tower.get("location", {})
            hero_dist_to_my_tower = self.cal_dist(hero_pos, ally_tower_pos)
        
        if enemy_hero and ally_tower:
            enemy_hero_pos = enemy_hero["actor_state"]["location"]
            ally_tower_pos = ally_tower.get("location", {})
            enemy_hero_dist_to_my_tower = self.cal_dist(enemy_hero_pos, ally_tower_pos)
        
        # 塔攻击范围（估算为8000）
        TOWER_ATTACK_RANGE = 8000.0
        
        # 英雄是否在塔攻击范围内
        is_hero_in_enemy_tower_range = 1.0 if hero_dist_to_enemy_tower < TOWER_ATTACK_RANGE else 0.0
        is_enemy_hero_in_my_tower_range = 1.0 if enemy_hero_dist_to_my_tower < TOWER_ATTACK_RANGE else 0.0
        
        # 塔是否正在攻击英雄
        is_enemy_tower_aggro_hero = 0.0
        is_my_tower_aggro_enemy_hero = 0.0
        
        my_hero_runtime_id = my_hero["actor_state"].get("runtime_id") if my_hero else None
        enemy_hero_runtime_id = enemy_hero["actor_state"].get("runtime_id") if enemy_hero else None
        
        for npc in frame_state.get("npc_states", []):
            attack_target = npc.get("attack_target", -1)
            
            # 检查敌方塔是否攻击我方英雄
            if (enemy_tower and npc.get("runtime_id") == enemy_tower.get("runtime_id") 
                and attack_target == my_hero_runtime_id):
                is_enemy_tower_aggro_hero = 1.0
            
            # 检查我方塔是否攻击敌方英雄
            if (ally_tower and npc.get("runtime_id") == ally_tower.get("runtime_id") 
                and attack_target == enemy_hero_runtime_id):
                is_my_tower_aggro_enemy_hero = 1.0
        
        # 塔血量威胁等级
        tower_threat_level = 0.0
        if is_enemy_tower_aggro_hero and enemy_tower:
            tower_dps = enemy_tower.get("phy_atk", 0) / max(1000, enemy_tower.get("atk_spd", 1000)) * 1000
            hero_max_hp = my_hero["actor_state"].get("max_hp", 1) if my_hero else 1
            tower_threat_level = min(tower_dps / hero_max_hp, 1.0)
        
        # 英雄攻击塔的效率
        hero_tower_damage_efficiency = 0.0
        if my_hero and enemy_tower:
            hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            tower_def = enemy_tower.get("phy_def", 0)
            actual_damage = max(1, hero_atk - tower_def * 0.5)
            hero_tower_damage_efficiency = actual_damage / max(1, enemy_tower.get("hp", 1))
        
        return [
            hero_dist_to_enemy_tower / 30000.0,  # 归一化
            hero_dist_to_my_tower / 30000.0,
            enemy_hero_dist_to_my_tower / 30000.0,
            is_hero_in_enemy_tower_range,
            is_enemy_hero_in_my_tower_range,
            is_enemy_tower_aggro_hero,
            is_my_tower_aggro_enemy_hero,
            tower_threat_level,
            hero_tower_damage_efficiency,
            0.5,  # 占位符
        ]

    def _encode_tower_minion_interaction_features(self, frame_state):
        """
        防御塔与兵线交互特征 (8维) - 塔兵交互
        """
        ally_tower = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        
        # 获取小兵列表
        ally_soldiers = []
        enemy_soldiers = []
        
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        TOWER_ATTACK_RANGE = 8000.0
        
        # 塔范围内小兵数量统计
        allied_minions_in_enemy_tower_range_count = 0
        enemy_minions_in_my_tower_range_count = 0
        
        if enemy_tower:
            enemy_tower_pos = enemy_tower.get("location", {})
            for soldier in ally_soldiers:
                soldier_pos = soldier.get("location", {})
                if self.cal_dist(soldier_pos, enemy_tower_pos) < TOWER_ATTACK_RANGE:
                    allied_minions_in_enemy_tower_range_count += 1
        
        if ally_tower:
            ally_tower_pos = ally_tower.get("location", {})
            for soldier in enemy_soldiers:
                soldier_pos = soldier.get("location", {})
                if self.cal_dist(soldier_pos, ally_tower_pos) < TOWER_ATTACK_RANGE:
                    enemy_minions_in_my_tower_range_count += 1
        
        # 后门保护机制
        enemy_tower_has_backdoor_protection = 0.0 if allied_minions_in_enemy_tower_range_count > 0 else 1.0
        my_tower_has_backdoor_protection = 0.0 if enemy_minions_in_my_tower_range_count > 0 else 1.0
        
        # 塔是否攻击炮车
        def _is_cannon(soldier):
            config_id = soldier.get("config_id", 0)
            return config_id in [101, 102, 103, 104, 105]
        
        is_enemy_tower_aggro_cannon = 0.0
        is_my_tower_aggro_cannon = 0.0
        
        # 获取炮车runtime_id
        ally_cannon_ids = [s.get("runtime_id") for s in ally_soldiers if _is_cannon(s)]
        enemy_cannon_ids = [s.get("runtime_id") for s in enemy_soldiers if _is_cannon(s)]
        
        for npc in frame_state.get("npc_states", []):
            attack_target = npc.get("attack_target", -1)
            
            if enemy_tower and npc.get("runtime_id") == enemy_tower.get("runtime_id"):
                if attack_target in ally_cannon_ids:
                    is_enemy_tower_aggro_cannon = 1.0
            
            if ally_tower and npc.get("runtime_id") == ally_tower.get("runtime_id"):
                if attack_target in enemy_cannon_ids:
                    is_my_tower_aggro_cannon = 1.0
        
        # 小兵对塔的DPS
        enemy_tower_dps_from_allied_minions = 0.0
        my_tower_dps_from_enemy_minions = 0.0
        
        if enemy_tower:
            enemy_tower_pos = enemy_tower.get("location", {})
            for soldier in ally_soldiers:
                soldier_pos = soldier.get("location", {})
                if self.cal_dist(soldier_pos, enemy_tower_pos) < TOWER_ATTACK_RANGE:
                    soldier_atk = soldier.get("phy_atk", 0) + soldier.get("mgc_atk", 0)
                    enemy_tower_dps_from_allied_minions += soldier_atk
        
        if ally_tower:
            ally_tower_pos = ally_tower.get("location", {})
            for soldier in enemy_soldiers:
                soldier_pos = soldier.get("location", {})
                if self.cal_dist(soldier_pos, ally_tower_pos) < TOWER_ATTACK_RANGE:
                    soldier_atk = soldier.get("phy_atk", 0) + soldier.get("mgc_atk", 0)
                    my_tower_dps_from_enemy_minions += soldier_atk
        
        return [
            allied_minions_in_enemy_tower_range_count / 15.0,  # 归一化
            enemy_minions_in_my_tower_range_count / 15.0,
            enemy_tower_has_backdoor_protection,
            my_tower_has_backdoor_protection,
            is_enemy_tower_aggro_cannon,
            is_my_tower_aggro_cannon,
            enemy_tower_dps_from_allied_minions / 3000.0,
            my_tower_dps_from_enemy_minions / 3000.0,
        ]

    def _encode_tower_predictive_features(self, frame_state):
        """
        防御塔预测与战略特征 (10维) - 预测特征 + 我的扩展
        """
        ally_tower = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        
        # 获取英雄信息
        my_hero = None
        enemy_hero = None
        if hasattr(self, 'main_hero_info') and self.main_hero_info:
            my_hero = self.main_hero_info
        
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            enemy_hero = hero
            break
        
        # 英雄单独摧毁塔的时间预测
        enemy_tower_time_to_destroy_by_hero = 999.0  # 秒
        my_tower_time_to_be_destroyed_by_hero = 999.0
        
        if my_hero and enemy_tower:
            hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            tower_hp = enemy_tower.get("hp", 0)
            tower_def = enemy_tower.get("phy_def", 0)
            actual_damage = max(1, hero_atk - tower_def * 0.5)
            hero_atk_speed = my_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
            attacks_per_second = 1000.0 / max(500, hero_atk_speed)
            
            if actual_damage > 0:
                enemy_tower_time_to_destroy_by_hero = tower_hp / (actual_damage * attacks_per_second)
        
        if enemy_hero and ally_tower:
            enemy_atk = enemy_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            tower_hp = ally_tower.get("hp", 0)
            tower_def = ally_tower.get("phy_def", 0)
            actual_damage = max(1, enemy_atk - tower_def * 0.5)
            enemy_atk_speed = enemy_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
            attacks_per_second = 1000.0 / max(500, enemy_atk_speed)
            
            if actual_damage > 0:
                my_tower_time_to_be_destroyed_by_hero = tower_hp / (actual_damage * attacks_per_second)
        
        # 塔血量危机等级
        tower_crisis_level = 0.0
        if ally_tower:
            my_tower_hp_ratio = ally_tower.get("hp", 0) / max(1, ally_tower.get("max_hp", 1))
            if my_tower_hp_ratio < 0.3:
                tower_crisis_level = 1.0 - my_tower_hp_ratio
        
        # 推塔窗口期评估
        push_tower_window = 0.0
        if enemy_tower and my_hero:
            ally_soldiers = []
            enemy_soldiers = []
            
            for npc in frame_state.get("npc_states", []):
                if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                    if npc.get("camp") == self.main_camp:
                        ally_soldiers.append(npc)
                    else:
                        enemy_soldiers.append(npc)
            
            minion_advantage = len(ally_soldiers) - len(enemy_soldiers)
            hero_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            
            if minion_advantage >= 2 and hero_hp_ratio > 0.6:
                push_tower_window = 0.8
            elif minion_advantage >= 0 and hero_hp_ratio > 0.4:
                push_tower_window = 0.5
            else:
                push_tower_window = 0.2
        
        # 回防紧迫性
        defend_urgency = 0.0
        if ally_tower and enemy_hero:
            enemy_hero_pos = enemy_hero["actor_state"]["location"]
            ally_tower_pos = ally_tower.get("location", {})
            distance_to_tower = self.cal_dist(enemy_hero_pos, ally_tower_pos)
            
            distance_factor = max(0, 1.0 - distance_to_tower / 15000.0)
            hp_factor = 1.0 - (ally_tower.get("hp", 0) / max(1, ally_tower.get("max_hp", 1)))
            defend_urgency = (distance_factor * 0.6 + hp_factor * 0.4)
        
        # 塔下团战优势
        tower_teamfight_advantage = 0.0
        if ally_tower and my_hero:
            ally_tower_pos = ally_tower.get("location", {})
            hero_pos = my_hero["actor_state"]["location"]
            hero_to_tower_dist = self.cal_dist(hero_pos, ally_tower_pos)
            
            if hero_to_tower_dist < 8000:
                tower_dps = ally_tower.get("phy_atk", 0) / max(1000, ally_tower.get("atk_spd", 1000)) * 1000
                hero_dps = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                total_dps = tower_dps + hero_dps
                
                enemy_hero_dps = 0
                if enemy_hero:
                    enemy_hero_dps = enemy_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                
                if total_dps > enemy_hero_dps:
                    tower_teamfight_advantage = min((total_dps - enemy_hero_dps) / total_dps, 1.0)
        
        # 塔攻击目标优先级合理性
        tower_target_priority_score = 0.5
        if ally_tower:
            for npc in frame_state.get("npc_states", []):
                if npc.get("runtime_id") == ally_tower.get("runtime_id"):
                    attack_target = npc.get("attack_target", -1)
                    if attack_target != -1:
                        if enemy_hero and attack_target == enemy_hero["actor_state"].get("runtime_id"):
                            tower_target_priority_score = 1.0
                        else:
                            tower_target_priority_score = 0.6
        
        # 小兵对塔的总DPS（重用前面的计算）
        ally_soldiers = []
        enemy_soldiers = []
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        enemy_tower_dps_from_allied_minions = 0.0
        my_tower_dps_from_enemy_minions = 0.0
        
        TOWER_ATTACK_RANGE = 8000.0
        
        if enemy_tower:
            enemy_tower_pos = enemy_tower.get("location", {})
            for soldier in ally_soldiers:
                soldier_pos = soldier.get("location", {})
                if self.cal_dist(soldier_pos, enemy_tower_pos) < TOWER_ATTACK_RANGE:
                    soldier_atk = soldier.get("phy_atk", 0) + soldier.get("mgc_atk", 0)
                    enemy_tower_dps_from_allied_minions += soldier_atk
        
        if ally_tower:
            ally_tower_pos = ally_tower.get("location", {})
            for soldier in enemy_soldiers:
                soldier_pos = soldier.get("location", {})
                if self.cal_dist(soldier_pos, ally_tower_pos) < TOWER_ATTACK_RANGE:
                    soldier_atk = soldier.get("phy_atk", 0) + soldier.get("mgc_atk", 0)
                    my_tower_dps_from_enemy_minions += soldier_atk
        
        return [
            min(enemy_tower_time_to_destroy_by_hero / 60.0, 1.0),  # 归一化到分钟
            min(my_tower_time_to_be_destroyed_by_hero / 60.0, 1.0),
            enemy_tower_dps_from_allied_minions / 3000.0,
            my_tower_dps_from_enemy_minions / 3000.0,
            tower_crisis_level,
            push_tower_window,
            defend_urgency,
            tower_teamfight_advantage,
            tower_target_priority_score,
            0.5,  # 战略价值评估（占位符）
        ]

    # =================== JUNGLE MONSTER FEATURES =================== #
    def _encode_jungle_monster_features(self, frame_state):
        """
        野怪特征工程 (40维) - 战略资源控制与机会成本分析
        包含: 野怪自身状态(8维) + 英雄交互特征(12维) + 战略机会成本特征(12维) + 高级博弈特征(8维)
        """
        # 获取野怪信息
        jungle_monster = self._find_jungle_monster(frame_state)
        
        # 获取英雄信息
        my_hero = None
        enemy_hero = None
        if hasattr(self, 'main_hero_info') and self.main_hero_info:
            my_hero = self.main_hero_info
        
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            enemy_hero = hero
            break
        
        # 计算各类特征
        jungle_features = []
        
        # 1. 野怪自身与时序特征 (8维)
        jungle_features.extend(self._encode_monster_intrinsic_features(jungle_monster, frame_state))
        
        # 2. 英雄与野怪交互特征 (12维)
        jungle_features.extend(self._encode_hero_monster_interaction_features(jungle_monster, my_hero, enemy_hero))
        
        # 3. 战略与机会成本特征 (12维)
        jungle_features.extend(self._encode_jungle_strategic_features(jungle_monster, my_hero, enemy_hero, frame_state))
        
        # 4. 高级博弈与时机特征 (8维) - 我的创新扩展
        jungle_features.extend(self._encode_jungle_advanced_features(jungle_monster, my_hero, enemy_hero, frame_state))
        
        return jungle_features

    def _find_jungle_monster(self, frame_state):
        """
        从frame_state中找到墨家机关道的核心野怪
        """
        # 检查多个可能的数据源
        possible_sources = ['monster_states', 'organ_states', 'npc_states']
        
        for source in possible_sources:
            entities = frame_state.get(source, [])
            for entity in entities:
                if self._is_jungle_monster(entity):
                    return entity
        
        return None

    def _is_jungle_monster(self, entity):
        """
        判断一个实体是否是关键野怪
        基于config_id、type、sub_type等字段判断
        """
        # 检查类型标识
        entity_type = entity.get("type", "")
        sub_type = entity.get("sub_type", "")
        config_id = entity.get("config_id", 0)
        
        # 野怪类型检查
        if entity_type == "ACTOR_TYPE_MONSTER":
            return True
        
        if sub_type in ["ACTOR_SUB_MONSTER", "MONSTER", "JUNGLE_MONSTER"]:
            return True
        
        # 基于config_id的特定野怪检查（需要根据实际游戏配置调整）
        # 假设墨家机关道的关键野怪config_id在500-600范围
        if 500 <= config_id <= 600:
            return True
        
        # 基于名称检查（如果有name字段）
        name = entity.get("name", "").lower()
        if any(keyword in name for keyword in ["jungle", "monster", "野怪", "中立"]):
            return True
        
        return False

    def _encode_monster_intrinsic_features(self, jungle_monster, frame_state):
        """
        野怪自身与时序特征 (8维)
        """
        # 基础存在性
        monster_is_alive = 1.0 if jungle_monster and jungle_monster.get("hp", 0) > 0 else 0.0
        
        # 血量状态
        monster_hp_ratio = 0.0
        monster_max_hp = 0.0
        monster_current_hp = 0.0
        
        if monster_is_alive:
            current_hp = jungle_monster.get("hp", 0)
            max_hp = jungle_monster.get("max_hp", 1)
            monster_hp_ratio = current_hp / max(1, max_hp)
            monster_max_hp = max_hp / 10000.0  # 归一化
            monster_current_hp = current_hp / 10000.0
        
        # 刷新时机预测（创新特征）
        current_frame = frame_state.get("frame_no", 0)
        monster_respawn_timer_ratio = 0.0
        monster_respawn_soon = 0.0
        
        if not monster_is_alive:
            # 假设野怪刷新间隔为30秒 = 1800帧（60FPS）
            RESPAWN_INTERVAL_FRAMES = 1800
            # 这里需要记录上次死亡时间，暂时用简化逻辑
            # 在实际实现中，需要在类中维护last_death_frame状态
            if hasattr(self, 'monster_last_death_frame'):
                frames_since_death = current_frame - self.monster_last_death_frame
                monster_respawn_timer_ratio = min(frames_since_death / RESPAWN_INTERVAL_FRAMES, 1.0)
                monster_respawn_soon = 1.0 if monster_respawn_timer_ratio > 0.8 else 0.0
        else:
            # 野怪存活时更新死亡时间记录
            if not hasattr(self, 'monster_was_alive_last_frame') or not self.monster_was_alive_last_frame:
                # 野怪刚刷新，重置死亡时间
                self.monster_last_death_frame = current_frame - 1800  # 假设刚刷新
            self.monster_was_alive_last_frame = True
        
        if not monster_is_alive and not hasattr(self, 'monster_was_alive_last_frame'):
            # 初始化状态
            self.monster_last_death_frame = current_frame - 900  # 假设中期状态
            self.monster_was_alive_last_frame = False
        elif monster_is_alive and hasattr(self, 'monster_was_alive_last_frame') and not self.monster_was_alive_last_frame:
            # 检测到野怪死亡
            self.monster_last_death_frame = current_frame
            self.monster_was_alive_last_frame = False
        
        # 野怪攻击力和经验价值
        monster_damage_potential = 0.0
        monster_value_score = 0.0
        
        if monster_is_alive:
            monster_atk = jungle_monster.get("phy_atk", 0) + jungle_monster.get("mgc_atk", 0)
            monster_damage_potential = monster_atk / 1000.0  # 归一化攻击力
            
            # 野怪价值评分（基于血量、经验、金币等）
            monster_exp = jungle_monster.get("exp_reward", 0)
            monster_gold = jungle_monster.get("gold_reward", 0)
            monster_value_score = (monster_exp + monster_gold * 2) / 1000.0  # 综合价值
        
        return [
            monster_is_alive,
            monster_hp_ratio,
            monster_max_hp,
            monster_current_hp,
            monster_respawn_timer_ratio,
            monster_respawn_soon,
            monster_damage_potential,
            monster_value_score,
        ]

    def _encode_hero_monster_interaction_features(self, jungle_monster, my_hero, enemy_hero):
        """
        英雄与野怪交互特征 (12维)
        """
        monster_is_alive = jungle_monster and jungle_monster.get("hp", 0) > 0
        
        # 距离计算
        hero_dist_to_monster = 50000.0  # 很大的默认值
        enemy_dist_to_monster = 50000.0
        dist_advantage_for_monster = 0.0
        
        if monster_is_alive and my_hero:
            hero_pos = my_hero["actor_state"]["location"]
            monster_pos = jungle_monster.get("location", {})
            hero_dist_to_monster = self.cal_dist(hero_pos, monster_pos)
        
        if monster_is_alive and enemy_hero:
            enemy_pos = enemy_hero["actor_state"]["location"]
            monster_pos = jungle_monster.get("location", {})
            enemy_dist_to_monster = self.cal_dist(enemy_pos, monster_pos)
        
        # 距离优势计算
        if monster_is_alive:
            dist_advantage_for_monster = enemy_dist_to_monster - hero_dist_to_monster
        
        # 攻击状态检查
        is_hero_attacking_monster = 0.0
        is_enemy_attacking_monster = 0.0
        
        if monster_is_alive:
            monster_runtime_id = jungle_monster.get("runtime_id")
            
            # 检查我方英雄是否在攻击野怪
            if my_hero:
                hero_target = my_hero["actor_state"].get("attack_target")
                if hero_target == monster_runtime_id:
                    is_hero_attacking_monster = 1.0
            
            # 检查敌方英雄是否在攻击野怪
            if enemy_hero:
                enemy_target = enemy_hero["actor_state"].get("attack_target")
                if enemy_target == monster_runtime_id:
                    is_enemy_attacking_monster = 1.0
        
        # 击杀时间预测
        time_to_kill_monster_by_hero = 999.0  # 很大的默认值（秒）
        time_to_kill_monster_by_enemy = 999.0
        
        if monster_is_alive:
            monster_hp = jungle_monster.get("hp", 0)
            
            # 我方击杀时间
            if my_hero:
                hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                hero_atk_speed = my_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
                attacks_per_second = 1000.0 / max(500, hero_atk_speed)
                
                if hero_atk > 0:
                    time_to_kill_monster_by_hero = monster_hp / (hero_atk * attacks_per_second)
            
            # 敌方击杀时间
            if enemy_hero:
                enemy_atk = enemy_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                enemy_atk_speed = enemy_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
                enemy_attacks_per_second = 1000.0 / max(500, enemy_atk_speed)
                
                if enemy_atk > 0:
                    time_to_kill_monster_by_enemy = monster_hp / (enemy_atk * enemy_attacks_per_second)
        
        # 野怪对英雄的威胁评估
        monster_threat_to_hero = 0.0
        monster_threat_to_enemy = 0.0
        
        if monster_is_alive:
            monster_atk = jungle_monster.get("phy_atk", 0) + jungle_monster.get("mgc_atk", 0)
            
            if my_hero and hero_dist_to_monster < 3000:  # 近距离威胁
                hero_hp = my_hero["actor_state"].get("hp", 1)
                monster_threat_to_hero = min(monster_atk / hero_hp, 1.0)
            
            if enemy_hero and enemy_dist_to_monster < 3000:
                enemy_hp = enemy_hero["actor_state"].get("hp", 1)
                monster_threat_to_enemy = min(monster_atk / enemy_hp, 1.0)
        
        # 抢夺竞争激烈度
        steal_competition_level = 0.0
        if monster_is_alive:
            # 基于双方距离差距和攻击力差距计算竞争激烈度
            distance_factor = max(0, 1.0 - abs(dist_advantage_for_monster) / 5000.0)
            
            hero_dps = 0
            enemy_dps = 0
            if my_hero:
                hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                hero_atk_speed = my_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
                hero_dps = hero_atk * 1000 / max(500, hero_atk_speed)
            
            if enemy_hero:
                enemy_atk = enemy_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                enemy_atk_speed = enemy_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
                enemy_dps = enemy_atk * 1000 / max(500, enemy_atk_speed)
            
            dps_factor = 1.0 - abs(hero_dps - enemy_dps) / max(hero_dps + enemy_dps, 1)
            steal_competition_level = (distance_factor * 0.6 + dps_factor * 0.4)
        
        return [
            hero_dist_to_monster / 50000.0,  # 归一化距离
            enemy_dist_to_monster / 50000.0,
            dist_advantage_for_monster / 20000.0,  # 归一化距离优势
            is_hero_attacking_monster,
            is_enemy_attacking_monster,
            min(time_to_kill_monster_by_hero / 30.0, 1.0),  # 归一化到30秒
            min(time_to_kill_monster_by_enemy / 30.0, 1.0),
            monster_threat_to_hero,
            monster_threat_to_enemy,
            steal_competition_level,
            0.5,  # 野怪攻击偏好（占位符）
            0.5,  # 击杀确定性（占位符）
        ]

    def _encode_jungle_strategic_features(self, jungle_monster, my_hero, enemy_hero, frame_state):
        """
        战略与机会成本特征 (12维)
        """
        monster_is_alive = jungle_monster and jungle_monster.get("hp", 0) > 0
        
        # 1. 兵线推进优势评估
        lane_push_advantage = 0.0
        lane_pressure_on_enemy = 0.0
        
        # 获取兵线信息计算推进优势
        ally_soldiers = []
        enemy_soldiers = []
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        if ally_soldiers and enemy_soldiers:
            # 计算兵线质心
            ally_center_z = sum(s.get("location", {}).get("z", 0) for s in ally_soldiers) / len(ally_soldiers)
            enemy_center_z = sum(s.get("location", {}).get("z", 0) for s in enemy_soldiers) / len(enemy_soldiers)
            
            # 转换为阵营坐标（己方在负Z）
            if self.transform_camp2_to_camp1:
                ally_center_z = -ally_center_z
                enemy_center_z = -enemy_center_z
            
            # 计算推进优势（正值表示我方兵线更靠前）
            lane_push_advantage = (enemy_center_z - ally_center_z) / 20000.0
            lane_push_advantage = max(-1.0, min(1.0, lane_push_advantage))
            
            # 计算对敌方塔的压力
            enemy_tower = self.enemy_camp_organ_dict.get("tower")
            if enemy_tower and ally_soldiers:
                min_dist_to_enemy_tower = min(
                    self.cal_dist(s.get("location", {}), enemy_tower.get("location", {}))
                    for s in ally_soldiers
                )
                lane_pressure_on_enemy = max(0, 1.0 - min_dist_to_enemy_tower / 15000.0)
        
        # 2. 敌方威胁评估
        enemy_threat_on_monster_contest = 1.0  # 默认高威胁
        enemy_hero_status_advantage = 0.0
        
        if enemy_hero:
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            enemy_dist_to_monster = 50000.0
            
            if monster_is_alive:
                enemy_pos = enemy_hero["actor_state"]["location"]
                monster_pos = jungle_monster.get("location", {})
                enemy_dist_to_monster = self.cal_dist(enemy_pos, monster_pos)
            
            # 威胁评估：距离远、血量低、死亡状态都降低威胁
            SAFE_DISTANCE = 8000.0
            if enemy_hp_ratio <= 0 or enemy_dist_to_monster > SAFE_DISTANCE:
                enemy_threat_on_monster_contest = 0.0
            elif enemy_hp_ratio < 0.3:
                enemy_threat_on_monster_contest = 0.3
            elif enemy_dist_to_monster > SAFE_DISTANCE * 0.5:
                enemy_threat_on_monster_contest = 0.5
            else:
                enemy_threat_on_monster_contest = 1.0
            
            # 我方相对敌方的状态优势
            if my_hero:
                my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
                enemy_hero_status_advantage = my_hp_ratio - enemy_hp_ratio
        
        # 3. Buff状态检查
        buff_is_active_on_hero = 0.0
        buff_is_active_on_enemy = 0.0
        
        # 检查双方是否已有野怪Buff（简化实现，实际需要根据具体buff ID检查）
        if my_hero:
            # 假设野怪Buff会增加攻击力或生命回复
            hero_buffs = my_hero.get("buff_state", {}).get("buffs", [])
            for buff in hero_buffs:
                # 检查是否是野怪提供的Buff（需要根据实际Buff ID调整）
                buff_id = buff.get("config_id", 0)
                if 1000 <= buff_id <= 1100:  # 假设的野怪Buff ID范围
                    buff_is_active_on_hero = 1.0
                    break
        
        if enemy_hero:
            enemy_buffs = enemy_hero.get("buff_state", {}).get("buffs", [])
            for buff in enemy_buffs:
                buff_id = buff.get("config_id", 0)
                if 1000 <= buff_id <= 1100:
                    buff_is_active_on_enemy = 1.0
                    break
        
        # 4. 机会成本计算 - 核心创新特征
        gold_loss_on_lane_while_killing_monster = 0.0
        exp_loss_on_lane_while_killing_monster = 0.0
        
        if monster_is_alive and my_hero:
            # 计算击杀野怪的时间成本
            monster_hp = jungle_monster.get("hp", 0)
            hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            hero_atk_speed = my_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
            
            if hero_atk > 0:
                time_to_kill = monster_hp / (hero_atk * 1000 / max(500, hero_atk_speed))
                
                # 估算在此时间内会漏掉的兵线收益
                # 假设每10秒刷新6个小兵，每个小兵15金币和20经验
                minions_missed = (time_to_kill / 10.0) * 6
                gold_loss_on_lane_while_killing_monster = minions_missed * 15
                exp_loss_on_lane_while_killing_monster = minions_missed * 20
        
        # 5. 塔压制窗口
        tower_pressure_window = 0.0
        if monster_is_alive:
            # 如果兵线优势+敌方低威胁，则是好的打野时机
            if lane_push_advantage > 0.2 and enemy_threat_on_monster_contest < 0.5:
                tower_pressure_window = 0.8
            elif lane_push_advantage > 0 and enemy_threat_on_monster_contest < 0.7:
                tower_pressure_window = 0.5
        
        # 6. 野怪价值相对性
        monster_relative_value = 0.5
        if monster_is_alive:
            # 基于当前游戏阶段和双方状态差距评估野怪的相对价值
            current_frame = frame_state.get("frame_no", 0)
            game_time = current_frame / 60.0  # 转换为秒
            
            # 游戏前期野怪价值更高（回复效果更重要）
            if game_time < 300:  # 5分钟前
                monster_relative_value = 0.8
            elif game_time < 600:  # 10分钟前
                monster_relative_value = 0.6
            else:
                monster_relative_value = 0.4
            
            # 如果血量劣势，野怪价值提升
            if my_hero and enemy_hero:
                my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
                enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
                
                if my_hp_ratio < enemy_hp_ratio - 0.2:
                    monster_relative_value += 0.3
        
        return [
            lane_push_advantage,
            lane_pressure_on_enemy,
            1.0 - enemy_threat_on_monster_contest,  # 转换为安全度
            enemy_hero_status_advantage,
            buff_is_active_on_hero,
            buff_is_active_on_enemy,
            min(gold_loss_on_lane_while_killing_monster / 200.0, 1.0),  # 归一化机会成本
            min(exp_loss_on_lane_while_killing_monster / 300.0, 1.0),
            tower_pressure_window,
            monster_relative_value,
            0.5,  # 团队收益vs个人收益（占位符）
            0.5,  # 长期战略价值（占位符）
        ]

    def _encode_jungle_advanced_features(self, jungle_monster, my_hero, enemy_hero, frame_state):
        """
        高级博弈与时机特征 (8维) - 我的创新扩展
        这些特征体现更深层的博弈论思维和预测能力
        """
        monster_is_alive = jungle_monster and jungle_monster.get("hp", 0) > 0
        
        # 1. 对手行为预测与反制
        enemy_likely_to_contest = 0.0
        our_counter_play_advantage = 0.0
        
        if monster_is_alive and enemy_hero:
            # 基于敌方状态预测其是否会来争夺野怪
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            enemy_pos = enemy_hero["actor_state"]["location"]
            monster_pos = jungle_monster.get("location", {})
            enemy_dist = self.cal_dist(enemy_pos, monster_pos)
            
            # 如果敌方血量健康且距离不远，可能会来争夺
            if enemy_hp_ratio > 0.6 and enemy_dist < 10000:
                enemy_likely_to_contest = 0.8
            elif enemy_hp_ratio > 0.4 and enemy_dist < 6000:
                enemy_likely_to_contest = 0.6
            else:
                enemy_likely_to_contest = 0.2
            
            # 我方的反制优势（基于技能、血量、位置）
            if my_hero:
                my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
                hero_pos = my_hero["actor_state"]["location"]
                my_dist = self.cal_dist(hero_pos, monster_pos)
                
                # 距离优势
                distance_advantage = max(0, (enemy_dist - my_dist) / 5000.0)
                # 血量优势
                hp_advantage = my_hp_ratio - enemy_hp_ratio
                # 技能优势（简化：基于EP状态）
                my_ep_ratio = my_hero["actor_state"].get("ep", 0) / max(1, my_hero["actor_state"].get("max_ep", 1))
                enemy_ep_ratio = enemy_hero["actor_state"].get("ep", 0) / max(1, enemy_hero["actor_state"].get("max_ep", 1))
                skill_advantage = my_ep_ratio - enemy_ep_ratio
                
                our_counter_play_advantage = (distance_advantage * 0.4 + hp_advantage * 0.4 + skill_advantage * 0.2)
                our_counter_play_advantage = max(-1.0, min(1.0, our_counter_play_advantage))
        
        # 2. 时机窗口精确预测
        optimal_timing_window = 0.0
        timing_pressure_level = 0.0
        
        if monster_is_alive:
            current_frame = frame_state.get("frame_no", 0)
            
            # 基于多种因素计算最优时机窗口
            factors = []
            
            # 兵线状态因子
            ally_soldiers = []
            enemy_soldiers = []
            for npc in frame_state.get("npc_states", []):
                if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                    if npc.get("camp") == self.main_camp:
                        ally_soldiers.append(npc)
                    else:
                        enemy_soldiers.append(npc)
            
            if len(ally_soldiers) > len(enemy_soldiers):
                factors.append(0.3)  # 兵线优势
            elif len(ally_soldiers) < len(enemy_soldiers):
                factors.append(-0.2)  # 兵线劣势
            else:
                factors.append(0.0)
            
            # 敌方威胁因子
            if enemy_hero:
                enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
                if enemy_hp_ratio < 0.3:
                    factors.append(0.4)  # 敌方低血
                elif enemy_hp_ratio > 0.8:
                    factors.append(-0.2)  # 敌方状态好
                else:
                    factors.append(0.0)
            
            # 自身状态因子
            if my_hero:
                my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
                if my_hp_ratio < 0.4:
                    factors.append(0.3)  # 我方需要回复
                else:
                    factors.append(0.0)
            
            optimal_timing_window = max(0, min(1.0, sum(factors) + 0.5))
            
            # 时机压力（即将错过的紧迫感）
            # 基于野怪刷新时间、敌方接近速度等
            if hasattr(self, 'monster_respawn_timer_ratio'):
                # 刚刷新的野怪，时机压力较低
                timing_pressure_level = 1.0 - self.monster_respawn_timer_ratio
            else:
                timing_pressure_level = 0.5
        
        # 3. 博弈心理与风险偏好
        psychological_pressure_on_enemy = 0.0
        risk_tolerance_adjustment = 0.5
        
        if monster_is_alive and my_hero and enemy_hero:
            # 基于双方状态差距产生的心理压力
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            
            # 如果我方优势明显，对敌方产生心理压力
            status_gap = my_hp_ratio - enemy_hp_ratio
            if status_gap > 0.3:
                psychological_pressure_on_enemy = 0.8
            elif status_gap > 0.1:
                psychological_pressure_on_enemy = 0.5
            else:
                psychological_pressure_on_enemy = 0.2
            
            # 风险偏好调整（血量低时更保守，优势时更激进）
            if my_hp_ratio > 0.7:
                risk_tolerance_adjustment = 0.8  # 高血量，可以冒险
            elif my_hp_ratio < 0.3:
                risk_tolerance_adjustment = 0.2  # 低血量，保守
            else:
                risk_tolerance_adjustment = 0.5
        
        # 4. 连锁反应预测
        chain_reaction_benefit = 0.0
        secondary_objective_opportunity = 0.0
        
        if monster_is_alive:
            # 击杀野怪后的连锁反应评估
            # 获得回复 -> 血量优势 -> 推塔机会
            if my_hero:
                my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
                
                # 如果当前血量低，野怪回复效果的连锁价值更高
                if my_hp_ratio < 0.5:
                    estimated_hp_after_monster = min(1.0, my_hp_ratio + 0.3)  # 假设野怪回复30%
                    
                    # 回复后是否能获得推塔/对拼优势
                    if enemy_hero:
                        enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
                        if estimated_hp_after_monster > enemy_hp_ratio + 0.2:
                            chain_reaction_benefit = 0.8  # 高连锁价值
                        else:
                            chain_reaction_benefit = 0.4
                
                # 次要目标机会（如击杀野怪后立即推塔）
                enemy_tower = self.enemy_camp_organ_dict.get("tower")
                if enemy_tower and monster_is_alive:
                    monster_pos = jungle_monster.get("location", {})
                    tower_pos = enemy_tower.get("location", {})
                    monster_to_tower_dist = self.cal_dist(monster_pos, tower_pos)
                    
                    # 如果野怪距离塔较近，击杀后可以顺势推塔
                    if monster_to_tower_dist < 8000:
                        secondary_objective_opportunity = 0.7
                    elif monster_to_tower_dist < 12000:
                        secondary_objective_opportunity = 0.4
                    else:
                        secondary_objective_opportunity = 0.1
        
        return [
            enemy_likely_to_contest,
            our_counter_play_advantage,
            optimal_timing_window,
            timing_pressure_level,
            psychological_pressure_on_enemy,
            risk_tolerance_adjustment,
            chain_reaction_benefit,
            secondary_objective_opportunity,
        ]

    # =================== ADVANCED TOWER FEATURES =================== #
    def _encode_advanced_tower_features(self, frame_state):
        """
        高级防御塔特征 (50维) - 动态风险评估与战略机会分析
        包含: 动态攻防风险评估(15维) + 战略机会交换评估(15维) + 微操时机特征(10维) + 心理博弈特征(10维)
        """
        # 获取基础数据
        ally_tower = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        
        my_hero = None
        enemy_hero = None
        if hasattr(self, 'main_hero_info') and self.main_hero_info:
            my_hero = self.main_hero_info
        
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            enemy_hero = hero
            break
        
        # 计算各类高级特征
        tower_features = []
        
        # 1. 动态攻防风险评估特征 (15维)
        tower_features.extend(self._encode_dynamic_risk_assessment(ally_tower, enemy_tower, my_hero, enemy_hero, frame_state))
        
        # 2. 战略机会与交换评估特征 (15维)
        tower_features.extend(self._encode_strategic_opportunity_assessment(ally_tower, enemy_tower, my_hero, enemy_hero, frame_state))
        
        # 3. 微操时机与执行特征 (10维) - 我的创新扩展
        tower_features.extend(self._encode_tower_micro_timing_features(ally_tower, enemy_tower, my_hero, enemy_hero, frame_state))
        
        # 4. 心理博弈与压制特征 (10维) - 我的创新扩展
        tower_features.extend(self._encode_tower_psychological_features(ally_tower, enemy_tower, my_hero, enemy_hero, frame_state))
        
        return tower_features

    def _encode_dynamic_risk_assessment(self, ally_tower, enemy_tower, my_hero, enemy_hero, frame_state):
        """
        动态攻防风险评估特征 (15维) - 你要求的核心风险量化
        """
        # 1. 我方塔生存时间预测
        my_tower_time_to_live_under_push = 999.0  # 秒
        
        if ally_tower and ally_tower.get("hp", 0) > 0:
            tower_hp = ally_tower.get("hp", 0)
            
            # 计算塔周围敌方单位的总DPS
            total_enemy_dps = 0.0
            TOWER_RANGE = 8000.0
            tower_pos = ally_tower.get("location", {})
            
            # 敌方英雄DPS
            if enemy_hero:
                enemy_pos = enemy_hero["actor_state"]["location"]
                enemy_dist = self.cal_dist(enemy_pos, tower_pos)
                if enemy_dist < TOWER_RANGE:
                    enemy_atk = enemy_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                    enemy_atk_speed = enemy_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
                    enemy_dps = enemy_atk * 1000 / max(500, enemy_atk_speed)
                    total_enemy_dps += enemy_dps
            
            # 敌方小兵DPS
            for npc in frame_state.get("npc_states", []):
                if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                    npc.get("camp") != self.main_camp and npc.get("hp", 0) > 0):
                    soldier_pos = npc.get("location", {})
                    soldier_dist = self.cal_dist(soldier_pos, tower_pos)
                    if soldier_dist < TOWER_RANGE:
                        soldier_atk = npc.get("phy_atk", 0) + npc.get("mgc_atk", 0)
                        total_enemy_dps += soldier_atk
            
            if total_enemy_dps > 1e-6:
                my_tower_time_to_live_under_push = tower_hp / total_enemy_dps
        
        # 2. 敌方英雄越塔威胁评分
        enemy_hero_dive_threat_score = 0.0
        
        if enemy_hero and ally_tower and my_hero:
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            
            # 统计敌方塔下小兵数量
            enemy_minions_count = 0
            ally_tower_pos = ally_tower.get("location", {})
            for npc in frame_state.get("npc_states", []):
                if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                    npc.get("camp") != self.main_camp and npc.get("hp", 0) > 0):
                    soldier_pos = npc.get("location", {})
                    if self.cal_dist(soldier_pos, ally_tower_pos) < 8000:
                        enemy_minions_count += 1
            
            # 威胁评分计算
            enemy_hero_dive_threat_score = (
                enemy_hp_ratio * (1 + enemy_minions_count * 0.2) / (my_hp_ratio + 0.1)
            )
            enemy_hero_dive_threat_score = min(enemy_hero_dive_threat_score, 3.0)
        
        # 3. 我方英雄塔下生存能力
        hero_tower_tanking_endurance_sec = 0.0
        tower_escape_time_sec = 999.0
        hero_dive_survivability_margin = -999.0
        
        if my_hero and enemy_tower:
            # 扛塔时间计算
            hero_hp = my_hero["actor_state"].get("hp", 0)
            tower_atk = enemy_tower.get("phy_atk", 0)
            
            # 简化护甲计算（实际应该根据具体护甲公式）
            hero_armor = my_hero["actor_state"].get("values", {}).get("phy_def", 0)
            armor_reduction = hero_armor / (hero_armor + 600)  # 简化护甲公式
            actual_tower_damage = tower_atk * (1 - armor_reduction)
            
            if actual_tower_damage > 0:
                tower_atk_speed = enemy_tower.get("atk_spd", 1000)
                tower_dps = actual_tower_damage * 1000 / max(500, tower_atk_speed)
                hero_tower_tanking_endurance_sec = hero_hp / tower_dps
            
            # 逃离时间计算
            hero_pos = my_hero["actor_state"]["location"]
            enemy_tower_pos = enemy_tower.get("location", {})
            current_dist = self.cal_dist(hero_pos, enemy_tower_pos)
            
            # 计算到塔攻击范围边缘的距离
            TOWER_ATTACK_RANGE = 8000.0
            escape_distance = max(0, TOWER_ATTACK_RANGE - current_dist)
            hero_move_speed = my_hero["actor_state"].get("values", {}).get("move_spd", 300)
            
            if hero_move_speed > 0:
                tower_escape_time_sec = escape_distance / hero_move_speed
            
            # 生存空间计算
            hero_dive_survivability_margin = hero_tower_tanking_endurance_sec - tower_escape_time_sec
        
        # 4. 塔攻击模式分析
        tower_attack_pattern_efficiency = 0.5
        tower_targeting_optimality = 0.5
        
        if enemy_tower:
            # 检查敌方塔的攻击模式和目标选择
            for npc in frame_state.get("npc_states", []):
                if npc.get("runtime_id") == enemy_tower.get("runtime_id"):
                    attack_target = npc.get("attack_target", -1)
                    
                    if attack_target != -1:
                        # 分析塔的目标选择是否最优
                        # 这里简化实现，实际应该检查目标的血量、威胁度等
                        tower_targeting_optimality = 0.7  # 假设一般情况下较优
                        tower_attack_pattern_efficiency = 0.8
        
        # 5. 我方越塔执行窗口
        ally_dive_execution_window = 0.0
        enemy_dive_execution_window = 0.0
        
        if my_hero and enemy_tower:
            # 基于英雄状态、小兵支援、敌方英雄位置等计算越塔窗口
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            
            # 统计我方塔下小兵
            ally_minions_near_enemy_tower = 0
            enemy_tower_pos = enemy_tower.get("location", {})
            for npc in frame_state.get("npc_states", []):
                if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                    npc.get("camp") == self.main_camp and npc.get("hp", 0) > 0):
                    soldier_pos = npc.get("location", {})
                    if self.cal_dist(soldier_pos, enemy_tower_pos) < 8000:
                        ally_minions_near_enemy_tower += 1
            
            # 敌方英雄威胁
            enemy_threat = 0.0
            if enemy_hero:
                enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
                enemy_pos = enemy_hero["actor_state"]["location"]
                enemy_to_tower_dist = self.cal_dist(enemy_pos, enemy_tower_pos)
                
                if enemy_to_tower_dist < 12000:  # 敌方英雄在附近
                    enemy_threat = enemy_hp_ratio
            
            # 综合计算越塔窗口
            if my_hp_ratio > 0.6 and ally_minions_near_enemy_tower >= 3 and enemy_threat < 0.4:
                ally_dive_execution_window = 0.8
            elif my_hp_ratio > 0.4 and ally_minions_near_enemy_tower >= 2 and enemy_threat < 0.6:
                ally_dive_execution_window = 0.5
            else:
                ally_dive_execution_window = 0.2
        
        # 同样计算敌方对我方塔的越塔窗口
        if enemy_hero and ally_tower:
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1)) if my_hero else 0
            
            # 统计敌方塔下小兵
            enemy_minions_near_ally_tower = 0
            ally_tower_pos = ally_tower.get("location", {})
            for npc in frame_state.get("npc_states", []):
                if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                    npc.get("camp") != self.main_camp and npc.get("hp", 0) > 0):
                    soldier_pos = npc.get("location", {})
                    if self.cal_dist(soldier_pos, ally_tower_pos) < 8000:
                        enemy_minions_near_ally_tower += 1
            
            if enemy_hp_ratio > 0.6 and enemy_minions_near_ally_tower >= 3 and my_hp_ratio < 0.4:
                enemy_dive_execution_window = 0.8
            elif enemy_hp_ratio > 0.4 and enemy_minions_near_ally_tower >= 2 and my_hp_ratio < 0.6:
                enemy_dive_execution_window = 0.5
            else:
                enemy_dive_execution_window = 0.2
        
        # 6. 塔下团战胜率预测
        tower_teamfight_win_probability = 0.5
        
        if ally_tower and my_hero and enemy_hero:
            # 综合评估塔下团战的胜率
            my_total_combat_power = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            enemy_total_combat_power = enemy_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            
            # 加上塔的DPS支援
            tower_dps = ally_tower.get("phy_atk", 0) / max(1000, ally_tower.get("atk_spd", 1000)) * 1000
            my_total_combat_power += tower_dps
            
            if my_total_combat_power + enemy_total_combat_power > 0:
                tower_teamfight_win_probability = my_total_combat_power / (my_total_combat_power + enemy_total_combat_power)
        
        return [
            min(my_tower_time_to_live_under_push / 60.0, 1.0),  # 归一化到分钟
            min(enemy_hero_dive_threat_score / 3.0, 1.0),  # 归一化威胁分数
            min(hero_tower_tanking_endurance_sec / 10.0, 1.0),  # 归一化到10秒
            min(tower_escape_time_sec / 5.0, 1.0),  # 归一化到5秒
            max(-1.0, min(hero_dive_survivability_margin / 10.0, 1.0)),  # 生存空间
            tower_attack_pattern_efficiency,
            tower_targeting_optimality,
            ally_dive_execution_window,
            enemy_dive_execution_window,
            tower_teamfight_win_probability,
            0.5,  # 塔攻击节奏（占位符）
            0.5,  # 塔仇恨转移效率（占位符）
            0.5,  # 塔下地形优势（占位符）
            0.5,  # 塔血量心理影响（占位符）
            0.5,  # 塔攻击威慑力（占位符）
        ]

    def _encode_strategic_opportunity_assessment(self, ally_tower, enemy_tower, my_hero, enemy_hero, frame_state):
        """
        战略机会与交换评估特征 (15维) - 你要求的战略交换分析
        """
        # 1. 塔伤害潜力计算
        tower_damage_potential_per_wave = 0.0
        
        if enemy_tower:
            # 计算敌方塔下我方小兵的总DPS
            TOWER_RANGE = 8000.0
            enemy_tower_pos = enemy_tower.get("location", {})
            allied_minions_dps = 0.0
            allied_minions_count = 0
            
            for npc in frame_state.get("npc_states", []):
                if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                    npc.get("camp") == self.main_camp and npc.get("hp", 0) > 0):
                    soldier_pos = npc.get("location", {})
                    if self.cal_dist(soldier_pos, enemy_tower_pos) < TOWER_RANGE:
                        soldier_atk = npc.get("phy_atk", 0) + npc.get("mgc_atk", 0)
                        allied_minions_dps += soldier_atk
                        allied_minions_count += 1
            
            # 假设一波兵线平均存活15秒
            WAVE_ALIVE_SECONDS = 15.0
            tower_damage_potential_per_wave = allied_minions_dps * WAVE_ALIVE_SECONDS
        
        # 2. 敌方塔血量变化追踪
        enemy_tower_hp_delta_last_10s = 0.0
        
        # 这里需要历史数据，简化实现
        # 实际应该维护一个历史血量记录
        if enemy_tower:
            current_hp = enemy_tower.get("hp", 0)
            # 简化：基于当前状态估算变化
            if hasattr(self, 'enemy_tower_last_hp'):
                enemy_tower_hp_delta_last_10s = self.enemy_tower_last_hp - current_hp
            self.enemy_tower_last_hp = current_hp
        
        # 3. 目标交换优势评分
        objective_trade_advantage_score = 0.0
        
        if enemy_tower:
            enemy_tower_hp_ratio = enemy_tower.get("hp", 0) / max(1, enemy_tower.get("max_hp", 1))
            tower_priority = 1.0 / max(0.1, enemy_tower_hp_ratio)  # 血量越低优先级越高
            
            # 与野怪价值对比
            jungle_monster = self._find_jungle_monster(frame_state)
            if jungle_monster and jungle_monster.get("hp", 0) > 0:
                monster_hp_ratio = jungle_monster.get("hp", 0) / max(1, jungle_monster.get("max_hp", 1))
                jungle_priority = 1.0 / max(0.1, monster_hp_ratio)
                
                objective_trade_advantage_score = tower_priority - jungle_priority
            else:
                objective_trade_advantage_score = tower_priority  # 野怪不存在，塔优先级最高
        
        # 4. 兵线时机控制
        time_since_last_wave_crashed = 0.0
        
        # 简化实现：基于当前兵线状态估算
        ally_soldiers = []
        for npc in frame_state.get("npc_states", []):
            if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                npc.get("camp") == self.main_camp and npc.get("hp", 0) > 0):
                ally_soldiers.append(npc)
        
        if len(ally_soldiers) == 0:
            # 可能刚清完一波
            time_since_last_wave_crashed = 1.0  # 标记为刚结束
        else:
            time_since_last_wave_crashed = 0.0
        
        # 5. 敌方塔目标选择优化度
        enemy_tower_target_is_optimal = 0.5
        
        if enemy_tower:
            for npc in frame_state.get("npc_states", []):
                if npc.get("runtime_id") == enemy_tower.get("runtime_id"):
                    attack_target = npc.get("attack_target", -1)
                    
                    if attack_target != -1:
                        # 检查攻击目标是否为威胁最大的单位
                        # 简化实现：检查是否攻击英雄或炮车
                        if my_hero and attack_target == my_hero["actor_state"].get("runtime_id"):
                            enemy_tower_target_is_optimal = 1.0  # 攻击英雄是最优的
                        else:
                            # 检查是否攻击炮车
                            target_is_cannon = False
                            for soldier_npc in frame_state.get("npc_states", []):
                                if (soldier_npc.get("runtime_id") == attack_target and
                                    soldier_npc.get("config_id", 0) in [101, 102, 103, 104, 105]):
                                    target_is_cannon = True
                                    break
                            
                            if target_is_cannon:
                                enemy_tower_target_is_optimal = 0.8  # 攻击炮车也不错
                            else:
                                enemy_tower_target_is_optimal = 0.4  # 攻击普通小兵不是最优
        
        # 6. 推塔节奏控制
        tower_push_rhythm_control = 0.5
        
        if enemy_tower and my_hero:
            # 基于兵线状态和英雄位置计算推塔节奏
            enemy_tower_pos = enemy_tower.get("location", {})
            hero_pos = my_hero["actor_state"]["location"]
            hero_to_tower_dist = self.cal_dist(hero_pos, enemy_tower_pos)
            
            # 统计塔下小兵数量
            minions_in_range = 0
            for npc in frame_state.get("npc_states", []):
                if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                    npc.get("camp") == self.main_camp and npc.get("hp", 0) > 0):
                    soldier_pos = npc.get("location", {})
                    if self.cal_dist(soldier_pos, enemy_tower_pos) < 8000:
                        minions_in_range += 1
            
            # 节奏评分：英雄距离 + 小兵数量
            distance_factor = max(0, 1.0 - hero_to_tower_dist / 12000.0)
            minion_factor = min(minions_in_range / 6.0, 1.0)
            tower_push_rhythm_control = (distance_factor * 0.6 + minion_factor * 0.4)
        
        # 7. 资源投入回报率
        tower_resource_investment_roi = 0.5
        
        if enemy_tower:
            # 计算攻击塔的投入成本和预期回报
            tower_hp_ratio = enemy_tower.get("hp", 0) / max(1, enemy_tower.get("max_hp", 1))
            
            # 投入：时间成本 + 血量成本
            time_cost = 0.0
            hp_cost = 0.0
            
            if my_hero:
                # 预估摧毁塔需要的时间
                hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                if hero_atk > 0:
                    time_to_destroy = enemy_tower.get("hp", 0) / hero_atk
                    time_cost = min(time_to_destroy / 30.0, 1.0)  # 归一化到30秒
                
                # 预估承受的塔伤害
                tower_atk = enemy_tower.get("phy_atk", 0)
                hero_hp = my_hero["actor_state"].get("hp", 0)
                if hero_hp > 0:
                    hp_cost = min((tower_atk * time_to_destroy) / hero_hp, 1.0)
            
            # 回报：塔的战略价值
            strategic_value = 1.0 - tower_hp_ratio  # 血量越低价值越高
            
            # ROI计算
            total_cost = (time_cost + hp_cost) / 2.0
            if total_cost > 0:
                tower_resource_investment_roi = strategic_value / total_cost
                tower_resource_investment_roi = min(tower_resource_investment_roi, 2.0) / 2.0
        
        # 8. 塔攻防转换时机
        tower_offense_defense_transition_timing = 0.5
        
        if ally_tower and enemy_tower and my_hero:
            # 基于双方塔血量和英雄状态判断攻防转换
            my_tower_hp_ratio = ally_tower.get("hp", 0) / max(1, ally_tower.get("max_hp", 1))
            enemy_tower_hp_ratio = enemy_tower.get("hp", 0) / max(1, enemy_tower.get("max_hp", 1))
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            
            # 如果我方塔血量危险，应该转为防守
            if my_tower_hp_ratio < 0.3:
                tower_offense_defense_transition_timing = 0.1  # 强烈建议防守
            elif my_tower_hp_ratio < 0.5 and my_hp_ratio < 0.4:
                tower_offense_defense_transition_timing = 0.3  # 建议防守
            elif enemy_tower_hp_ratio < 0.3 and my_hp_ratio > 0.6:
                tower_offense_defense_transition_timing = 0.9  # 强烈建议进攻
            elif enemy_tower_hp_ratio < 0.5:
                tower_offense_defense_transition_timing = 0.7  # 建议进攻
            else:
                tower_offense_defense_transition_timing = 0.5  # 平衡状态
        
        # 9. 塔周围地图控制
        tower_map_control_advantage = 0.5
        
        if enemy_tower:
            # 基于塔周围的兵线分布和英雄位置计算地图控制优势
            enemy_tower_pos = enemy_tower.get("location", {})
            
            # 统计塔周围的兵力分布
            ally_units_near_tower = 0
            enemy_units_near_tower = 0
            
            CONTROL_RANGE = 12000.0
            
            # 统计小兵
            for npc in frame_state.get("npc_states", []):
                if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                    soldier_pos = npc.get("location", {})
                    dist_to_tower = self.cal_dist(soldier_pos, enemy_tower_pos)
                    
                    if dist_to_tower < CONTROL_RANGE:
                        if npc.get("camp") == self.main_camp:
                            ally_units_near_tower += 1
                        else:
                            enemy_units_near_tower += 1
            
            # 统计英雄
            if my_hero:
                hero_pos = my_hero["actor_state"]["location"]
                if self.cal_dist(hero_pos, enemy_tower_pos) < CONTROL_RANGE:
                    ally_units_near_tower += 3  # 英雄权重更高
            
            if enemy_hero:
                enemy_pos = enemy_hero["actor_state"]["location"]
                if self.cal_dist(enemy_pos, enemy_tower_pos) < CONTROL_RANGE:
                    enemy_units_near_tower += 3
            
            # 地图控制优势
            total_units = ally_units_near_tower + enemy_units_near_tower
            if total_units > 0:
                tower_map_control_advantage = ally_units_near_tower / total_units
        
        # 10. 塔血量阈值决策
        tower_hp_threshold_decision = 0.5
        
        if enemy_tower:
            enemy_tower_hp_ratio = enemy_tower.get("hp", 0) / max(1, enemy_tower.get("max_hp", 1))
            
            # 基于塔血量给出决策倾向
            if enemy_tower_hp_ratio < 0.1:
                tower_hp_threshold_decision = 1.0  # 必须推掉
            elif enemy_tower_hp_ratio < 0.2:
                tower_hp_threshold_decision = 0.9  # 强烈建议推
            elif enemy_tower_hp_ratio < 0.3:
                tower_hp_threshold_decision = 0.7  # 建议推
            elif enemy_tower_hp_ratio < 0.5:
                tower_hp_threshold_decision = 0.6  # 可以考虑推
            else:
                tower_hp_threshold_decision = 0.3  # 优先级不高
        
        return [
            min(tower_damage_potential_per_wave / 5000.0, 1.0),  # 归一化兵线伤害潜力
            max(-1.0, min(enemy_tower_hp_delta_last_10s / 1000.0, 1.0)),  # 归一化血量变化
            max(-2.0, min(objective_trade_advantage_score / 2.0, 2.0)) / 2.0 + 0.5,  # 归一化交换优势
            time_since_last_wave_crashed,
            enemy_tower_target_is_optimal,
            tower_push_rhythm_control,
            tower_resource_investment_roi,
            tower_offense_defense_transition_timing,
            tower_map_control_advantage,
            tower_hp_threshold_decision,
            0.5,  # 塔经济价值预期（占位符）
            0.5,  # 塔时间窗口优化（占位符）
            0.5,  # 塔风险收益平衡（占位符）
            0.5,  # 塔协同作战价值（占位符）
            0.5,  # 塔长期战略意义（占位符）
        ]

    def _encode_tower_micro_timing_features(self, ally_tower, enemy_tower, my_hero, enemy_hero, frame_state):
        """
        微操时机与执行特征 (10维) - 我的创新扩展
        这些特征专注于精确的时机把握和微操执行
        """
        # 1. 塔攻击间隔利用
        tower_attack_cooldown_exploitation = 0.0
        
        if enemy_tower:
            # 基于塔的攻击速度计算攻击间隔
            tower_atk_speed = enemy_tower.get("atk_spd", 1000)  # ms
            attack_interval = tower_atk_speed / 1000.0  # 秒
            
            # 评估在攻击间隔内可以进行的操作
            if my_hero:
                hero_move_speed = my_hero["actor_state"].get("values", {}).get("move_spd", 300)
                # 在塔攻击间隔内能移动的距离
                movement_in_interval = hero_move_speed * attack_interval
                
                # 利用评分：移动距离越大，利用价值越高
                tower_attack_cooldown_exploitation = min(movement_in_interval / 1000.0, 1.0)
        
        # 2. 仇恨转移时机
        aggro_transfer_timing = 0.0
        
        if enemy_tower and my_hero:
            # 检查是否有小兵可以承担塔的仇恨
            enemy_tower_pos = enemy_tower.get("location", {})
            hero_pos = my_hero["actor_state"]["location"]
            hero_to_tower_dist = self.cal_dist(hero_pos, enemy_tower_pos)
            
            # 寻找塔下的我方小兵
            minions_in_tower_range = []
            for npc in frame_state.get("npc_states", []):
                if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                    npc.get("camp") == self.main_camp and npc.get("hp", 0) > 0):
                    soldier_pos = npc.get("location", {})
                    dist_to_tower = self.cal_dist(soldier_pos, enemy_tower_pos)
                    if dist_to_tower < 8000:
                        minions_in_tower_range.append(npc)
            
            # 如果有小兵在塔下，仇恨转移时机较好
            if len(minions_in_tower_range) > 0 and hero_to_tower_dist < 8000:
                # 找到最近的小兵
                closest_minion_dist = min(
                    self.cal_dist(m.get("location", {}), hero_pos) 
                    for m in minions_in_tower_range
                )
                
                # 仇恨转移时机评分
                if closest_minion_dist < 2000:  # 小兵很近
                    aggro_transfer_timing = 0.9
                elif closest_minion_dist < 4000:
                    aggro_transfer_timing = 0.7
                else:
                    aggro_transfer_timing = 0.4
        
        # 3. 技能释放窗口优化
        skill_cast_window_optimization = 0.0
        
        if my_hero and enemy_tower:
            # 基于技能冷却和塔攻击时机优化技能释放
            hero_skills = my_hero.get("skill_state", {}).get("skills", [])
            available_skills = 0
            
            for skill in hero_skills:
                if skill.get("usable", False):
                    available_skills += 1
            
            # 如果有可用技能，评估释放时机
            if available_skills > 0:
                hero_pos = my_hero["actor_state"]["location"]
                enemy_tower_pos = enemy_tower.get("location", {})
                dist_to_tower = self.cal_dist(hero_pos, enemy_tower_pos)
                
                # 基于距离和技能数量评估
                distance_factor = max(0, 1.0 - dist_to_tower / 10000.0)
                skill_factor = min(available_skills / 4.0, 1.0)
                skill_cast_window_optimization = (distance_factor * 0.6 + skill_factor * 0.4)
        
        # 4. 走砍节奏优化
        kiting_rhythm_optimization = 0.0
        
        if my_hero and enemy_hero:
            # 计算走砍的最优节奏
            my_atk_speed = my_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
            my_move_speed = my_hero["actor_state"].get("values", {}).get("move_spd", 300)
            
            enemy_atk_speed = enemy_hero["actor_state"].get("values", {}).get("atk_spd", 1000)
            enemy_move_speed = enemy_hero["actor_state"].get("values", {}).get("move_spd", 300)
            
            # 攻击间隔内的移动优势
            my_movement_per_attack = my_move_speed * (my_atk_speed / 1000.0)
            enemy_movement_per_attack = enemy_move_speed * (enemy_atk_speed / 1000.0)
            
            if my_movement_per_attack > 0:
                kiting_advantage = my_movement_per_attack / (my_movement_per_attack + enemy_movement_per_attack)
                kiting_rhythm_optimization = kiting_advantage
        
        # 5. 塔下补刀时机
        tower_last_hit_timing = 0.0
        
        if enemy_tower:
            # 计算在塔下补刀的最佳时机
            enemy_tower_pos = enemy_tower.get("location", {})
            tower_atk = enemy_tower.get("phy_atk", 0)
            
            # 寻找塔下的敌方小兵
            enemy_minions_under_tower = []
            for npc in frame_state.get("npc_states", []):
                if (npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and 
                    npc.get("camp") != self.main_camp and npc.get("hp", 0) > 0):
                    soldier_pos = npc.get("location", {})
                    if self.cal_dist(soldier_pos, enemy_tower_pos) < 8000:
                        enemy_minions_under_tower.append(npc)
            
            # 评估补刀时机
            if enemy_minions_under_tower and my_hero:
                hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                
                # 找到可以被击杀的小兵
                killable_minions = 0
                for minion in enemy_minions_under_tower:
                    minion_hp = minion.get("hp", 0)
                    # 考虑塔的攻击，预测小兵血量
                    predicted_hp = minion_hp - tower_atk  # 简化预测
                    
                    if predicted_hp > 0 and predicted_hp < hero_atk:
                        killable_minions += 1
                
                if killable_minions > 0:
                    tower_last_hit_timing = min(killable_minions / 3.0, 1.0)
        
        # 6. 逃跑路径预规划
        escape_path_pre_planning = 0.0
        
        if my_hero and enemy_tower:
            hero_pos = my_hero["actor_state"]["location"]
            enemy_tower_pos = enemy_tower.get("location", {})
            
            # 计算多个可能的逃跑路径
            escape_routes = []
            
            # 简化：计算8个方向的逃跑路径
            import math
            for angle in range(0, 360, 45):
                radian = math.radians(angle)
                # 逃跑方向的终点
                escape_x = hero_pos.get("x", 0) + 5000 * math.cos(radian)
                escape_z = hero_pos.get("z", 0) + 5000 * math.sin(radian)
                
                # 计算这个方向距离塔的距离
                escape_to_tower_dist = math.sqrt(
                    (escape_x - enemy_tower_pos.get("x", 0))**2 + 
                    (escape_z - enemy_tower_pos.get("z", 0))**2
                )
                
                escape_routes.append(escape_to_tower_dist)
            
            # 最优逃跑路径评分
            if escape_routes:
                max_escape_dist = max(escape_routes)
                min_escape_dist = min(escape_routes)
                
                # 路径选择多样性和安全性评分
                path_diversity = (max_escape_dist - min_escape_dist) / max(max_escape_dist, 1)
                path_safety = min_escape_dist / 15000.0  # 最近路径的安全性
                
                escape_path_pre_planning = (path_diversity * 0.4 + min(path_safety, 1.0) * 0.6)
        
        # 7. 动态风险阈值调整
        dynamic_risk_threshold_adjustment = 0.5
        
        if my_hero:
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            
            # 基于血量动态调整风险承受能力
            if my_hp_ratio > 0.8:
                dynamic_risk_threshold_adjustment = 0.9  # 高血量，可以冒险
            elif my_hp_ratio > 0.6:
                dynamic_risk_threshold_adjustment = 0.7
            elif my_hp_ratio > 0.4:
                dynamic_risk_threshold_adjustment = 0.5
            elif my_hp_ratio > 0.2:
                dynamic_risk_threshold_adjustment = 0.3
            else:
                dynamic_risk_threshold_adjustment = 0.1  # 低血量，极度保守
        
        # 8. 连击执行窗口
        combo_execution_window = 0.0
        
        if my_hero and enemy_hero:
            # 评估执行连击的时机窗口
            hero_pos = my_hero["actor_state"]["location"]
            enemy_pos = enemy_hero["actor_state"]["location"]
            distance = self.cal_dist(hero_pos, enemy_pos)
            
            # 技能可用性检查
            hero_skills = my_hero.get("skill_state", {}).get("skills", [])
            usable_skills = sum(1 for skill in hero_skills if skill.get("usable", False))
            
            # 敌方状态检查
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            
            # 连击窗口评分
            distance_factor = max(0, 1.0 - distance / 8000.0)  # 距离因子
            skill_factor = min(usable_skills / 3.0, 1.0)  # 技能因子
            target_vulnerability = 1.0 - enemy_hp_ratio  # 目标脆弱性
            
            combo_execution_window = (distance_factor * 0.4 + skill_factor * 0.3 + target_vulnerability * 0.3)
        
        return [
            tower_attack_cooldown_exploitation,
            aggro_transfer_timing,
            skill_cast_window_optimization,
            kiting_rhythm_optimization,
            tower_last_hit_timing,
            escape_path_pre_planning,
            dynamic_risk_threshold_adjustment,
            combo_execution_window,
            0.5,  # 塔攻击预判（占位符）
            0.5,  # 微操流畅度（占位符）
        ]

    def _encode_tower_psychological_features(self, ally_tower, enemy_tower, my_hero, enemy_hero, frame_state):
        """
        心理博弈与压制特征 (10维) - 我的创新扩展  
        这些特征模拟心理层面的博弈和压制效果
        """
        # 1. 塔下压制效应
        tower_suppression_effect = 0.0
        
        if ally_tower and enemy_hero:
            # 计算我方塔对敌方英雄的心理压制
            ally_tower_pos = ally_tower.get("location", {})
            enemy_pos = enemy_hero["actor_state"]["location"]
            enemy_to_tower_dist = self.cal_dist(enemy_pos, ally_tower_pos)
            
            # 距离越近，压制效应越强
            if enemy_to_tower_dist < 8000:  # 塔攻击范围内
                tower_suppression_effect = 0.9
            elif enemy_to_tower_dist < 12000:  # 接近塔
                tower_suppression_effect = 0.6
            elif enemy_to_tower_dist < 16000:  # 塔的威慑范围
                tower_suppression_effect = 0.3
            else:
                tower_suppression_effect = 0.0
        
        # 2. 血量心理优势
        hp_psychological_advantage = 0.0
        
        if my_hero and enemy_hero:
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            
            hp_gap = my_hp_ratio - enemy_hp_ratio
            
            # 血量优势带来的心理优势
            if hp_gap > 0.4:
                hp_psychological_advantage = 0.9  # 巨大优势
            elif hp_gap > 0.2:
                hp_psychological_advantage = 0.7  # 明显优势
            elif hp_gap > 0:
                hp_psychological_advantage = 0.6  # 轻微优势
            elif hp_gap > -0.2:
                hp_psychological_advantage = 0.4  # 轻微劣势
            else:
                hp_psychological_advantage = 0.1  # 明显劣势
        
        # 3. 装备优势威慑
        equipment_intimidation_factor = 0.5
        
        if my_hero and enemy_hero:
            # 基于攻击力差距评估装备威慑
            my_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            enemy_atk = enemy_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            
            if my_atk > 0 and enemy_atk > 0:
                atk_ratio = my_atk / enemy_atk
                
                if atk_ratio > 1.5:
                    equipment_intimidation_factor = 0.9  # 装备碾压
                elif atk_ratio > 1.2:
                    equipment_intimidation_factor = 0.7  # 装备优势
                elif atk_ratio > 0.8:
                    equipment_intimidation_factor = 0.5  # 旗鼓相当
                else:
                    equipment_intimidation_factor = 0.3  # 装备劣势
        
        # 4. 主动权控制感
        initiative_control_perception = 0.5
        
        if my_hero and enemy_hero and enemy_tower:
            # 基于位置和行为模式评估主动权
            hero_pos = my_hero["actor_state"]["location"]
            enemy_pos = enemy_hero["actor_state"]["location"]
            enemy_tower_pos = enemy_tower.get("location", {})
            
            # 计算谁更接近敌方目标（推塔）
            my_dist_to_enemy_tower = self.cal_dist(hero_pos, enemy_tower_pos)
            enemy_dist_to_my_pos = self.cal_dist(enemy_pos, hero_pos)
            
            # 如果我更接近敌方塔，说明我在主导节奏
            if my_dist_to_enemy_tower < 12000 and enemy_dist_to_my_pos > 8000:
                initiative_control_perception = 0.8  # 我在主导推进
            elif my_dist_to_enemy_tower < enemy_dist_to_my_pos:
                initiative_control_perception = 0.6  # 我比较主动
            else:
                initiative_control_perception = 0.4  # 敌方比较主动
        
        # 5. 领域控制感
        territorial_control_feeling = 0.5
        
        if ally_tower and enemy_tower:
            # 基于双方塔血量差距评估领域控制感
            my_tower_hp_ratio = ally_tower.get("hp", 0) / max(1, ally_tower.get("max_hp", 1))
            enemy_tower_hp_ratio = enemy_tower.get("hp", 0) / max(1, enemy_tower.get("max_hp", 1))
            
            tower_hp_advantage = my_tower_hp_ratio - enemy_tower_hp_ratio
            
            # 塔血量优势带来的领域控制感
            if tower_hp_advantage > 0.3:
                territorial_control_feeling = 0.8
            elif tower_hp_advantage > 0.1:
                territorial_control_feeling = 0.6
            elif tower_hp_advantage > -0.1:
                territorial_control_feeling = 0.5
            elif tower_hp_advantage > -0.3:
                territorial_control_feeling = 0.4
            else:
                territorial_control_feeling = 0.2
        
        # 6. 时间压力感知
        time_pressure_perception = 0.5
        
        # 基于游戏进程评估时间压力
        current_frame = frame_state.get("frame_no", 0)
        game_duration = current_frame / 60.0  # 秒
        
        # 游戏时间越长，推塔压力越大
        if game_duration > 600:  # 10分钟后
            time_pressure_perception = 0.9
        elif game_duration > 300:  # 5分钟后
            time_pressure_perception = 0.7
        else:
            time_pressure_perception = 0.3
        
        # 7. 连杀威慑效应
        killing_spree_intimidation = 0.5
        
        if my_hero and enemy_hero:
            # 简化实现：基于英雄等级差距推测击杀情况
            my_level = my_hero.get("level", 1)
            enemy_level = enemy_hero.get("level", 1)
            
            level_gap = my_level - enemy_level
            
            if level_gap >= 3:
                killing_spree_intimidation = 0.9  # 巨大等级优势
            elif level_gap >= 2:
                killing_spree_intimidation = 0.7
            elif level_gap >= 1:
                killing_spree_intimidation = 0.6
            elif level_gap == 0:
                killing_spree_intimidation = 0.5
            else:
                killing_spree_intimidation = 0.3
        
        # 8. 反杀威胁恐惧
        counter_kill_threat_fear = 0.5
        
        if my_hero and enemy_hero:
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            
            # 当双方都是低血量时，反杀威胁最大
            if my_hp_ratio < 0.3 and enemy_hp_ratio < 0.3:
                counter_kill_threat_fear = 0.9  # 极高反杀风险
            elif my_hp_ratio < 0.5 and enemy_hp_ratio > 0.7:
                counter_kill_threat_fear = 0.8  # 我方劣势明显
            elif my_hp_ratio > 0.7 and enemy_hp_ratio < 0.3:
                counter_kill_threat_fear = 0.2  # 我方优势明显
            else:
                counter_kill_threat_fear = 0.5
        
        # 9. 技能威慑价值
        skill_intimidation_value = 0.5
        
        if my_hero:
            # 基于技能可用性评估威慑价值
            hero_skills = my_hero.get("skill_state", {}).get("skills", [])
            available_skills = sum(1 for skill in hero_skills if skill.get("usable", False))
            
            # 可用技能越多，威慑价值越高
            skill_intimidation_value = min(available_skills / 4.0, 1.0)
            
            # 如果大招可用，威慑价值额外提升
            if len(hero_skills) > 3 and hero_skills[3].get("usable", False):
                skill_intimidation_value = min(skill_intimidation_value + 0.3, 1.0)
        
        # 10. 逆转潜力评估
        comeback_potential_assessment = 0.5
        
        if my_hero and enemy_hero and ally_tower and enemy_tower:
            # 综合评估逆转比赛的潜力
            my_hp_ratio = my_hero["actor_state"].get("hp", 0) / max(1, my_hero["actor_state"].get("max_hp", 1))
            enemy_hp_ratio = enemy_hero["actor_state"].get("hp", 0) / max(1, enemy_hero["actor_state"].get("max_hp", 1))
            
            my_tower_hp_ratio = ally_tower.get("hp", 0) / max(1, ally_tower.get("max_hp", 1))
            enemy_tower_hp_ratio = enemy_tower.get("hp", 0) / max(1, enemy_tower.get("max_hp", 1))
            
            # 计算整体劣势程度
            hp_disadvantage = enemy_hp_ratio - my_hp_ratio
            tower_disadvantage = enemy_tower_hp_ratio - my_tower_hp_ratio
            
            total_disadvantage = (hp_disadvantage + tower_disadvantage) / 2.0
            
            # 劣势越大，逆转潜力的价值越高（但实现难度也越大）
            if total_disadvantage > 0.3:
                comeback_potential_assessment = 0.8  # 大逆转机会
            elif total_disadvantage > 0.1:
                comeback_potential_assessment = 0.6  # 小逆转机会
            elif total_disadvantage < -0.1:
                comeback_potential_assessment = 0.3  # 我方优势，逆转潜力低
            else:
                comeback_potential_assessment = 0.5  # 均势
        
        return [
            tower_suppression_effect,
            hp_psychological_advantage,
            equipment_intimidation_factor,
            initiative_control_perception,
            territorial_control_feeling,
            time_pressure_perception,
            killing_spree_intimidation,
            counter_kill_threat_fear,
            skill_intimidation_value,
            comeback_potential_assessment,
        ]

    # =================== EXPERT WAVE FEATURES =================== #
    def _encode_expert_wave_features(self, frame_state):
        """
        专家级兵线特征 (60维) - 兵线健康度分析与高级控制预测
        包含: 兵线健康度构成特征(15维) + 高级兵线控制预测(15维) + 兵线时序与节奏特征(15维) + 兵线心理与战术特征(15维)
        """
        # 获取基础数据
        ally_tower = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        
        my_hero = None
        enemy_hero = None
        if hasattr(self, 'main_hero_info') and self.main_hero_info:
            my_hero = self.main_hero_info
        
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            enemy_hero = hero
            break
        
        # 获取兵线数据
        ally_soldiers = []
        enemy_soldiers = []
        
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc.get("hp", 0) > 0:
                if npc.get("camp") == self.main_camp:
                    ally_soldiers.append(npc)
                else:
                    enemy_soldiers.append(npc)
        
        # 计算各类专家级特征
        wave_features = []
        
        # 1. 兵线健康度与构成细化特征 (15维) - 你要求的核心特征
        wave_features.extend(self._encode_wave_health_composition(ally_soldiers, enemy_soldiers, my_hero, enemy_hero, frame_state))
        
        # 2. 高级兵线控制与预测特征 (15维) - 你要求的核心特征
        wave_features.extend(self._encode_advanced_wave_control(ally_soldiers, enemy_soldiers, my_hero, enemy_hero, ally_tower, enemy_tower, frame_state))
        
        # 3. 兵线时序与节奏控制特征 (15维) - 我的创新扩展
        wave_features.extend(self._encode_wave_timing_rhythm(ally_soldiers, enemy_soldiers, my_hero, enemy_hero, frame_state))
        
        # 4. 兵线心理与战术博弈特征 (15维) - 我的创新扩展
        wave_features.extend(self._encode_wave_psychological_tactics(ally_soldiers, enemy_soldiers, my_hero, enemy_hero, frame_state))
        
        return wave_features

    def _encode_wave_health_composition(self, ally_soldiers, enemy_soldiers, my_hero, enemy_hero, frame_state):
        """
        兵线健康度与构成细化特征 (15维) - 你要求的精确补刀和清兵分析
        """
        # 1. 敌方兵线一刀击杀数量 - 你的核心需求
        enemy_wave_one_shot_killable_count = 0
        
        if my_hero:
            hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            
            for soldier in enemy_soldiers:
                soldier_hp = soldier.get("hp", 0)
                if soldier_hp > 0 and soldier_hp <= hero_atk:
                    enemy_wave_one_shot_killable_count += 1
        
        # 2. 敌方兵线一技能击杀数量 - 你的核心需求
        enemy_wave_one_skill_killable_count = 0
        
        if my_hero:
            # 简化实现：估算一技能伤害
            hero_skills = my_hero.get("skill_state", {}).get("skills", [])
            if hero_skills and len(hero_skills) > 0:
                # 基于英雄攻击力和等级估算技能伤害
                hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
                hero_level = my_hero.get("level", 1)
                skill_1_damage = hero_atk * 1.5 + hero_level * 20  # 简化的技能伤害公式
                
                for soldier in enemy_soldiers:
                    soldier_hp = soldier.get("hp", 0)
                    if soldier_hp > 0 and soldier_hp <= skill_1_damage:
                        enemy_wave_one_skill_killable_count += 1
        
        # 3. 敌方兵线血量分布 (3维) - 你的核心需求
        enemy_wave_hp_low = 0      # 0-33%
        enemy_wave_hp_medium = 0   # 33-66%
        enemy_wave_hp_high = 0     # 66-100%
        
        for soldier in enemy_soldiers:
            if soldier.get("hp", 0) > 0:
                hp_ratio = soldier.get("hp", 0) / max(1, soldier.get("max_hp", 1))
                
                if hp_ratio <= 0.33:
                    enemy_wave_hp_low += 1
                elif hp_ratio <= 0.66:
                    enemy_wave_hp_medium += 1
                else:
                    enemy_wave_hp_high += 1
        
        # 4. 我方兵线血量分布 (3维) - 你的核心需求
        allied_wave_hp_low = 0
        allied_wave_hp_medium = 0
        allied_wave_hp_high = 0
        
        for soldier in ally_soldiers:
            if soldier.get("hp", 0) > 0:
                hp_ratio = soldier.get("hp", 0) / max(1, soldier.get("max_hp", 1))
                
                if hp_ratio <= 0.33:
                    allied_wave_hp_low += 1
                elif hp_ratio <= 0.66:
                    allied_wave_hp_medium += 1
                else:
                    allied_wave_hp_high += 1
        
        # 5. 远程兵优势 - 你的核心需求
        wave_ranged_minion_advantage = 0
        
        ally_ranged_count = 0
        enemy_ranged_count = 0
        
        for soldier in ally_soldiers:
            # 基于configId判断是否为远程兵（通常包含archer等标识）
            config_id = soldier.get("config_id", 0)
            if self._is_ranged_minion(config_id):
                ally_ranged_count += 1
        
        for soldier in enemy_soldiers:
            config_id = soldier.get("config_id", 0)
            if self._is_ranged_minion(config_id):
                enemy_ranged_count += 1
        
        wave_ranged_minion_advantage = ally_ranged_count - enemy_ranged_count
        
        # 6. 炮车兵健康状况分析
        ally_cannon_hp_ratio = 0.0
        enemy_cannon_hp_ratio = 0.0
        
        for soldier in ally_soldiers:
            if self._is_cannon_minion(soldier.get("config_id", 0)):
                if soldier.get("hp", 0) > 0:
                    ally_cannon_hp_ratio = soldier.get("hp", 0) / max(1, soldier.get("max_hp", 1))
                break
        
        for soldier in enemy_soldiers:
            if self._is_cannon_minion(soldier.get("config_id", 0)):
                if soldier.get("hp", 0) > 0:
                    enemy_cannon_hp_ratio = soldier.get("hp", 0) / max(1, soldier.get("max_hp", 1))
                break
        
        # 7. 兵线整体生存预期
        allied_wave_survival_expectancy = 0.5
        enemy_wave_survival_expectancy = 0.5
        
        if ally_soldiers:
            total_hp = sum(s.get("hp", 0) for s in ally_soldiers)
            max_total_hp = sum(s.get("max_hp", 1) for s in ally_soldiers)
            allied_wave_survival_expectancy = total_hp / max(max_total_hp, 1)
        
        if enemy_soldiers:
            total_hp = sum(s.get("hp", 0) for s in enemy_soldiers)
            max_total_hp = sum(s.get("max_hp", 1) for s in enemy_soldiers)
            enemy_wave_survival_expectancy = total_hp / max(max_total_hp, 1)
        
        # 8. AOE清线效率评估
        aoe_clear_efficiency = 0.5
        
        if enemy_soldiers and my_hero:
            # 计算敌方兵线的聚集度
            if len(enemy_soldiers) > 1:
                positions = []
                for soldier in enemy_soldiers:
                    pos = soldier.get("location", {})
                    positions.append((pos.get("x", 0), pos.get("z", 0)))
                
                # 计算平均距离
                total_distance = 0
                count = 0
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        dist = ((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)**0.5
                        total_distance += dist
                        count += 1
                
                if count > 0:
                    avg_distance = total_distance / count
                    # 距离越小，AOE效率越高
                    aoe_clear_efficiency = max(0, 1.0 - avg_distance / 3000.0)
        
        return [
            min(enemy_wave_one_shot_killable_count / 6.0, 1.0),  # 归一化到最多6个小兵
            min(enemy_wave_one_skill_killable_count / 6.0, 1.0),
            min(enemy_wave_hp_low / 6.0, 1.0),  # 血量分布归一化
            min(enemy_wave_hp_medium / 6.0, 1.0),
            min(enemy_wave_hp_high / 6.0, 1.0),
            min(allied_wave_hp_low / 6.0, 1.0),
            min(allied_wave_hp_medium / 6.0, 1.0),
            min(allied_wave_hp_high / 6.0, 1.0),
            max(-3.0, min(wave_ranged_minion_advantage, 3.0)) / 3.0 + 0.5,  # 归一化到[-3,3]范围
            ally_cannon_hp_ratio,
            enemy_cannon_hp_ratio,
            allied_wave_survival_expectancy,
            enemy_wave_survival_expectancy,
            aoe_clear_efficiency,
            0.5,  # 兵线伤害类型分布（占位符）
        ]

    def _is_ranged_minion(self, config_id):
        """判断是否为远程兵"""
        # 这里需要根据实际游戏数据调整
        # 通常远程兵的config_id会有特定的范围或标识
        ranged_minion_ids = [102, 105, 108, 111, 114]  # 示例ID，需要根据实际调整
        return config_id in ranged_minion_ids

    def _is_cannon_minion(self, config_id):
        """判断是否为炮车兵"""
        # 炮车兵通常有特定的config_id
        cannon_minion_ids = [103, 106, 109, 112, 115]  # 示例ID，需要根据实际调整
        return config_id in cannon_minion_ids

    def _encode_advanced_wave_control(self, ally_soldiers, enemy_soldiers, my_hero, enemy_hero, ally_tower, enemy_tower, frame_state):
        """
        高级兵线控制与预测特征 (15维) - 你要求的控线、慢推、回推战术
        """
        # 1. 兵线控制潜力评分 - 你的核心需求
        wave_freeze_potential_score = 0.0
        
        if ally_tower and enemy_soldiers and ally_soldiers:
            # 计算交战点
            if enemy_soldiers and ally_soldiers:
                enemy_frontline_pos = self._get_frontline_position(enemy_soldiers, "forward")
                ally_tower_pos = ally_tower.get("location", {})
                
                clash_dist_to_my_tower = self.cal_dist(enemy_frontline_pos, ally_tower_pos)
                
                # 计算远程兵优势
                ally_ranged = sum(1 for s in ally_soldiers if self._is_ranged_minion(s.get("config_id", 0)))
                enemy_ranged = sum(1 for s in enemy_soldiers if self._is_ranged_minion(s.get("config_id", 0)))
                ranged_advantage = ally_ranged - enemy_ranged
                
                # 兵线数量对比
                minion_disadvantage = len(enemy_soldiers) - len(ally_soldiers)
                
                # 控线潜力公式：敌方兵多 + 距离我方塔近 + 我方远程兵优势
                if minion_disadvantage > 0 and clash_dist_to_my_tower < 15000:
                    distance_factor = max(0, 1.0 - clash_dist_to_my_tower / 15000.0)
                    ranged_factor = max(0, ranged_advantage) * 0.2
                    minion_factor = min(minion_disadvantage / 3.0, 1.0)
                    
                    wave_freeze_potential_score = (distance_factor * 0.5 + minion_factor * 0.3 + ranged_factor * 0.2)
        
        # 2. 兵线回推计时器 - 你的核心需求
        wave_bounce_back_timer = 0.0
        
        if enemy_tower and ally_soldiers:
            # 检查是否有我方小兵在敌方塔下
            enemy_tower_pos = enemy_tower.get("location", {})
            allied_minions_in_enemy_tower = 0
            
            for soldier in ally_soldiers:
                soldier_pos = soldier.get("location", {})
                if self.cal_dist(soldier_pos, enemy_tower_pos) < 8000:
                    allied_minions_in_enemy_tower += 1
            
            # 如果有兵在塔下，开始回推计时
            if allied_minions_in_enemy_tower > 0:
                # 简化实现：基于兵线数量估算回推时间
                # 实际应该维护一个时间追踪器
                if not hasattr(self, 'bounce_back_start_time'):
                    self.bounce_back_start_time = frame_state.get("frame_no", 0)
                
                current_frame = frame_state.get("frame_no", 0)
                elapsed_frames = current_frame - self.bounce_back_start_time
                
                # 估算回推到我方塔需要的时间（假设30秒）
                estimated_bounce_time = 30 * 60  # 30秒 * 60帧/秒
                wave_bounce_back_timer = min(elapsed_frames / estimated_bounce_time, 1.0)
            else:
                if hasattr(self, 'bounce_back_start_time'):
                    delattr(self, 'bounce_back_start_time')
        
        # 3. 下一波兵线到达时间比例 - 你的核心需求
        next_wave_arrival_time_ratio = 0.0
        
        # 简化实现：基于游戏时间估算兵线刷新
        current_frame = frame_state.get("frame_no", 0)
        WAVE_SPAWN_INTERVAL = 30 * 60  # 30秒间隔
        
        time_since_last_spawn = current_frame % WAVE_SPAWN_INTERVAL
        next_wave_arrival_time_ratio = time_since_last_spawn / WAVE_SPAWN_INTERVAL
        
        # 4. 屯兵线潜力评分 - 你的核心需求
        stacked_wave_potential_score = 0.0
        
        typical_wave_size = 3  # 标准一波兵线数量
        current_allied_count = len(ally_soldiers)
        
        if current_allied_count > typical_wave_size:
            stacked_wave_potential_score = min((current_allied_count - typical_wave_size) / 6.0, 1.0)
        
        # 5. 英雄对控线威胁 - 你的核心需求
        hero_threat_to_freeze = 0.0
        
        if enemy_hero and my_hero:
            # 检查敌方英雄是否在线上
            hero_pos = my_hero["actor_state"]["location"]
            enemy_pos = enemy_hero["actor_state"]["location"]
            
            hero_distance = self.cal_dist(hero_pos, enemy_pos)
            
            # 如果敌方英雄在附近
            if hero_distance < 8000:
                is_hero_in_lane = 1.0
                
                # 评估敌方英雄的AOE清线能力
                enemy_skills = enemy_hero.get("skill_state", {}).get("skills", [])
                aoe_clear_ability = 0.0
                
                # 简化实现：基于可用技能数量估算AOE能力
                available_skills = sum(1 for skill in enemy_skills if skill.get("usable", False))
                aoe_clear_ability = min(available_skills / 4.0, 1.0)
                
                hero_threat_to_freeze = is_hero_in_lane * aoe_clear_ability
        
        # 6. 兵线推进速度预测
        wave_push_speed_prediction = 0.5
        
        if ally_soldiers and enemy_soldiers:
            # 计算双方兵线的总DPS
            ally_total_dps = sum(s.get("phy_atk", 0) + s.get("mgc_atk", 0) for s in ally_soldiers)
            enemy_total_dps = sum(s.get("phy_atk", 0) + s.get("mgc_atk", 0) for s in enemy_soldiers)
            
            if ally_total_dps + enemy_total_dps > 0:
                # DPS优势决定推进方向和速度
                dps_advantage = (ally_total_dps - enemy_total_dps) / (ally_total_dps + enemy_total_dps)
                wave_push_speed_prediction = (dps_advantage + 1.0) / 2.0  # 归一化到[0,1]
        
        # 7. 慢推兵线识别
        slow_push_identification = 0.0
        
        if ally_soldiers and enemy_soldiers:
            # 慢推的特征：轻微的兵力优势，特别是远程兵优势
            ally_count = len(ally_soldiers)
            enemy_count = len(enemy_soldiers)
            
            ally_ranged = sum(1 for s in ally_soldiers if self._is_ranged_minion(s.get("config_id", 0)))
            enemy_ranged = sum(1 for s in enemy_soldiers if self._is_ranged_minion(s.get("config_id", 0)))
            
            # 轻微数量优势 + 远程兵优势 = 慢推
            if 0 < ally_count - enemy_count <= 2 and ally_ranged > enemy_ranged:
                slow_push_identification = 0.8
            elif ally_ranged > enemy_ranged:
                slow_push_identification = 0.5
        
        # 8. 兵线重置时机
        wave_reset_timing = 0.5
        
        # 基于兵线位置判断是否需要重置
        if ally_tower and enemy_tower and ally_soldiers:
            ally_tower_pos = ally_tower.get("location", {})
            enemy_tower_pos = enemy_tower.get("location", {})
            
            # 计算兵线质心
            ally_centroid = self._calculate_wave_centroid(ally_soldiers)
            
            # 如果兵线过度偏向一方，可能需要重置
            dist_to_ally_tower = self.cal_dist(ally_centroid, ally_tower_pos)
            dist_to_enemy_tower = self.cal_dist(ally_centroid, enemy_tower_pos)
            
            total_dist = dist_to_ally_tower + dist_to_enemy_tower
            if total_dist > 0:
                position_bias = abs(dist_to_ally_tower - dist_to_enemy_tower) / total_dist
                wave_reset_timing = position_bias  # 偏移越大，重置需求越高
        
        # 9. 兵线分割可能性
        wave_split_possibility = 0.0
        
        if ally_soldiers:
            # 检查兵线是否可能被分割（部分兵线脱节）
            positions = []
            for soldier in ally_soldiers:
                pos = soldier.get("location", {})
                positions.append((pos.get("x", 0), pos.get("z", 0)))
            
            if len(positions) > 2:
                # 计算兵线的最大跨度
                max_distance = 0
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        dist = ((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)**0.5
                        max_distance = max(max_distance, dist)
                
                # 跨度越大，分割可能性越高
                wave_split_possibility = min(max_distance / 5000.0, 1.0)
        
        return [
            wave_freeze_potential_score,
            wave_bounce_back_timer,
            next_wave_arrival_time_ratio,
            stacked_wave_potential_score,
            hero_threat_to_freeze,
            wave_push_speed_prediction,
            slow_push_identification,
            wave_reset_timing,
            wave_split_possibility,
            0.5,  # 快推识别（占位符）
            0.5,  # 兵线汇合预测（占位符）
            0.5,  # 兵线质量评估（占位符）
            0.5,  # 兵线控制难度（占位符）
            0.5,  # 兵线价值密度（占位符）
            0.5,  # 兵线同步性（占位符）
        ]

    def _get_frontline_position(self, soldiers, direction):
        """获取兵线最前沿位置"""
        if not soldiers:
            return {"x": 0, "z": 0}
        
        # 简化实现：返回第一个小兵的位置
        # 实际应该根据direction和地图特征计算真正的最前沿
        return soldiers[0].get("location", {"x": 0, "z": 0})

    def _calculate_wave_centroid(self, soldiers):
        """计算兵线质心"""
        if not soldiers:
            return {"x": 0, "z": 0}
        
        total_x = sum(s.get("location", {}).get("x", 0) for s in soldiers)
        total_z = sum(s.get("location", {}).get("z", 0) for s in soldiers)
        
        count = len(soldiers)
        return {
            "x": total_x / count,
            "z": total_z / count
        }

    def _encode_wave_timing_rhythm(self, ally_soldiers, enemy_soldiers, my_hero, enemy_hero, frame_state):
        """
        兵线时序与节奏控制特征 (15维) - 我的创新扩展
        这些特征专注于兵线的时间维度和节奏掌控
        """
        # 1. 兵线生命周期阶段
        wave_lifecycle_stage = 0.5
        
        if ally_soldiers:
            # 基于兵线血量分布判断生命周期阶段
            total_hp = sum(s.get("hp", 0) for s in ally_soldiers)
            max_total_hp = sum(s.get("max_hp", 1) for s in ally_soldiers)
            
            if max_total_hp > 0:
                hp_ratio = total_hp / max_total_hp
                
                if hp_ratio > 0.8:
                    wave_lifecycle_stage = 0.1  # 初期：满血状态
                elif hp_ratio > 0.6:
                    wave_lifecycle_stage = 0.3  # 早期：轻微损伤
                elif hp_ratio > 0.4:
                    wave_lifecycle_stage = 0.5  # 中期：中等损伤
                elif hp_ratio > 0.2:
                    wave_lifecycle_stage = 0.7  # 后期：重度损伤
                else:
                    wave_lifecycle_stage = 0.9  # 末期：濒临死亡
        
        # 2. 兵线节拍器（Metronome）
        wave_metronome_sync = 0.5
        
        # 基于双方兵线的攻击同步性
        if ally_soldiers and enemy_soldiers:
            # 简化实现：基于兵线数量和类型的同步性
            ally_count = len(ally_soldiers)
            enemy_count = len(enemy_soldiers)
            
            # 数量越接近，节拍越同步
            count_diff = abs(ally_count - enemy_count)
            wave_metronome_sync = max(0, 1.0 - count_diff / 6.0)
        
        # 3. 兵线呼吸节奏（推进-回撤循环）
        wave_breathing_rhythm = 0.5
        
        # 基于历史位置变化计算兵线的推进-回撤模式
        current_frame = frame_state.get("frame_no", 0)
        
        # 简化实现：使用正弦波模拟自然的推进-回撤节奏
        import math
        cycle_length = 120  # 2分钟一个周期
        phase = (current_frame % cycle_length) / cycle_length * 2 * math.pi
        wave_breathing_rhythm = (math.sin(phase) + 1.0) / 2.0
        
        # 4. 补刀窗口节奏
        last_hit_window_rhythm = 0.0
        
        if enemy_soldiers and my_hero:
            hero_atk = my_hero["actor_state"].get("values", {}).get("phy_atk", 0)
            
            # 计算即将进入补刀窗口的小兵数量
            near_last_hit_count = 0
            for soldier in enemy_soldiers:
                soldier_hp = soldier.get("hp", 0)
                # 考虑塔的伤害，预测小兵何时进入补刀范围
                if hero_atk * 1.2 >= soldier_hp >= hero_atk * 0.8:
                    near_last_hit_count += 1
            
            last_hit_window_rhythm = min(near_last_hit_count / 3.0, 1.0)
        
        # 5. 兵线聚合时机
        wave_convergence_timing = 0.5
        
        # 检查是否即将有新兵线与现有兵线汇合
        current_frame = frame_state.get("frame_no", 0)
        WAVE_SPAWN_INTERVAL = 30 * 60  # 30秒
        
        frames_until_next_wave = WAVE_SPAWN_INTERVAL - (current_frame % WAVE_SPAWN_INTERVAL)
        convergence_threshold = 10 * 60  # 10秒内
        
        if frames_until_next_wave <= convergence_threshold:
            wave_convergence_timing = 1.0 - (frames_until_next_wave / convergence_threshold)
        
        # 6. 兵线能量建立
        wave_energy_buildup = 0.0
        
        if ally_soldiers:
            # 基于兵线数量和类型计算"能量"
            total_units = len(ally_soldiers)
            cannon_count = sum(1 for s in ally_soldiers if self._is_cannon_minion(s.get("config_id", 0)))
            ranged_count = sum(1 for s in ally_soldiers if self._is_ranged_minion(s.get("config_id", 0)))
            
            # 能量公式：基础数量 + 炮车权重 + 远程兵权重
            energy_score = total_units + cannon_count * 2 + ranged_count * 1.5
            wave_energy_buildup = min(energy_score / 20.0, 1.0)  # 归一化
        
        # 7. 兵线衰减预测
        wave_decay_prediction = 0.5
        
        if ally_soldiers and enemy_soldiers:
            # 预测我方兵线相对于敌方的衰减速度
            ally_avg_hp = sum(s.get("hp", 0) for s in ally_soldiers) / len(ally_soldiers)
            enemy_avg_hp = sum(s.get("hp", 0) for s in enemy_soldiers) / len(enemy_soldiers)
            
            if ally_avg_hp + enemy_avg_hp > 0:
                # 血量比例越低，衰减越快
                hp_ratio = ally_avg_hp / (ally_avg_hp + enemy_avg_hp)
                wave_decay_prediction = 1.0 - hp_ratio  # 我方血量越低，衰减预测越高
        
        # 8. 兵线投入产出比时序
        wave_roi_timing = 0.5
        
        # 基于当前时间判断投入兵线资源的性价比
        current_frame = frame_state.get("frame_no", 0)
        game_duration = current_frame / 60.0  # 游戏时长（秒）
        
        # 游戏前期：兵线价值高；中期：英雄作用增强；后期：推塔急迫
        if game_duration < 300:  # 5分钟前
            wave_roi_timing = 0.8  # 前期兵线价值高
        elif game_duration < 600:  # 5-10分钟
            wave_roi_timing = 0.5  # 中期平衡
        else:  # 10分钟后
            wave_roi_timing = 0.9  # 后期推塔急迫，兵线重要
        
        # 9. 兵线冲击波预测
        wave_impact_prediction = 0.0
        
        if ally_soldiers:
            # 预测兵线到达关键位置（如塔下）的冲击力
            total_damage = sum(s.get("phy_atk", 0) + s.get("mgc_atk", 0) for s in ally_soldiers)
            total_hp = sum(s.get("hp", 0) for s in ally_soldiers)
            
            # 冲击力 = 伤害 × 生存能力
            impact_score = (total_damage * total_hp) / 10000.0
            wave_impact_prediction = min(impact_score, 1.0)
        
        # 10. 兵线同步性评估
        wave_synchronization_assessment = 0.5
        
        if ally_soldiers and len(ally_soldiers) > 1:
            # 评估兵线内部的移动同步性
            positions = []
            for soldier in ally_soldiers:
                pos = soldier.get("location", {})
                positions.append((pos.get("x", 0), pos.get("z", 0)))
            
            # 计算位置方差，方差越小，同步性越好
            if len(positions) > 1:
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_z = sum(p[1] for p in positions) / len(positions)
                
                variance = sum((p[0] - avg_x)**2 + (p[1] - avg_z)**2 for p in positions) / len(positions)
                
                # 方差越小，同步性越好
                wave_synchronization_assessment = max(0, 1.0 - variance / 5000000.0)
        
        return [
            wave_lifecycle_stage,
            wave_metronome_sync,
            wave_breathing_rhythm,
            last_hit_window_rhythm,
            wave_convergence_timing,
            wave_energy_buildup,
            wave_decay_prediction,
            wave_roi_timing,
            wave_impact_prediction,
            wave_synchronization_assessment,
            0.5,  # 兵线加速度（占位符）
            0.5,  # 兵线惯性（占位符）
            0.5,  # 兵线共振频率（占位符）
            0.5,  # 兵线时间价值（占位符）
            0.5,  # 兵线节拍偏差（占位符）
        ]

    def _encode_wave_psychological_tactics(self, ally_soldiers, enemy_soldiers, my_hero, enemy_hero, frame_state):
        """
        兵线心理与战术博弈特征 (15维) - 我的创新扩展
        这些特征模拟兵线运营中的心理博弈和战术欺骗
        """
        # 1. 兵线威慑效应
        wave_intimidation_effect = 0.0
        
        if ally_soldiers:
            # 基于兵线规模和构成计算威慑效应
            total_count = len(ally_soldiers)
            cannon_count = sum(1 for s in ally_soldiers if self._is_cannon_minion(s.get("config_id", 0)))
            
            # 兵线规模越大，威慑效应越强
            scale_factor = min(total_count / 6.0, 1.0)
            cannon_factor = min(cannon_count * 0.3, 0.6)
            
            wave_intimidation_effect = scale_factor + cannon_factor
            wave_intimidation_effect = min(wave_intimidation_effect, 1.0)
        
        # 2. 兵线虚实博弈
        wave_deception_potential = 0.5
        
        if ally_soldiers and enemy_soldiers:
            # 评估通过兵线操作进行战术欺骗的潜力
            ally_count = len(ally_soldiers)
            enemy_count = len(enemy_soldiers)
            
            # 当双方兵线实力接近时，虚实博弈空间最大
            count_balance = 1.0 - abs(ally_count - enemy_count) / max(ally_count + enemy_count, 1)
            
            # 远程兵的存在增加虚实操作的可能性
            ally_ranged = sum(1 for s in ally_soldiers if self._is_ranged_minion(s.get("config_id", 0)))
            ranged_factor = min(ally_ranged / 2.0, 0.5)
            
            wave_deception_potential = count_balance * 0.7 + ranged_factor * 0.3
        
        # 3. 兵线压制感知
        wave_pressure_perception = 0.5
        
        if enemy_soldiers and ally_soldiers:
            # 基于敌方兵线对我方的压制程度
            enemy_total_hp = sum(s.get("hp", 0) for s in enemy_soldiers)
            ally_total_hp = sum(s.get("hp", 0) for s in ally_soldiers)
            
            if ally_total_hp + enemy_total_hp > 0:
                pressure_ratio = enemy_total_hp / (enemy_total_hp + ally_total_hp)
                
                # 敌方兵线健康度越高，压制感越强
                if pressure_ratio > 0.7:
                    wave_pressure_perception = 0.9  # 强烈压制
                elif pressure_ratio > 0.6:
                    wave_pressure_perception = 0.7  # 中等压制
                elif pressure_ratio > 0.4:
                    wave_pressure_perception = 0.5  # 平衡状态
                else:
                    wave_pressure_perception = 0.3  # 我方优势
        
        # 4. 兵线心理优势
        wave_psychological_advantage = 0.5
        
        if ally_soldiers and enemy_soldiers:
            # 综合评估兵线给予的心理优势
            # 考虑数量、质量、位置等因素
            
            # 数量优势
            count_advantage = len(ally_soldiers) - len(enemy_soldiers)
            count_factor = max(-1.0, min(count_advantage / 3.0, 1.0))
            
            # 质量优势（炮车、远程兵）
            ally_cannon = sum(1 for s in ally_soldiers if self._is_cannon_minion(s.get("config_id", 0)))
            enemy_cannon = sum(1 for s in enemy_soldiers if self._is_cannon_minion(s.get("config_id", 0)))
            quality_advantage = ally_cannon - enemy_cannon
            quality_factor = max(-1.0, min(quality_advantage, 1.0))
            
            # 综合心理优势
            wave_psychological_advantage = (count_factor * 0.6 + quality_factor * 0.4 + 1.0) / 2.0
        
        # 5. 兵线操控难度
        wave_manipulation_difficulty = 0.5
        
        if ally_soldiers:
            # 评估精确操控当前兵线的难度
            spread_factor = 0.5
            
            if len(ally_soldiers) > 1:
                # 计算兵线的分散程度
                positions = []
                for soldier in ally_soldiers:
                    pos = soldier.get("location", {})
                    positions.append((pos.get("x", 0), pos.get("z", 0)))
                
                # 计算平均距离
                total_distance = 0
                count = 0
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        dist = ((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)**0.5
                        total_distance += dist
                        count += 1
                
                if count > 0:
                    avg_distance = total_distance / count
                    # 距离越大，操控难度越高
                    spread_factor = min(avg_distance / 3000.0, 1.0)
            
            # 兵线数量越多，操控也越难
            count_factor = min(len(ally_soldiers) / 8.0, 1.0)
            
            wave_manipulation_difficulty = spread_factor * 0.6 + count_factor * 0.4
        
        # 6. 兵线战术价值
        wave_tactical_value = 0.5
        
        if ally_soldiers:
            # 评估当前兵线的战术价值
            tactical_score = 0.0
            
            # 炮车的战术价值最高
            cannon_count = sum(1 for s in ally_soldiers if self._is_cannon_minion(s.get("config_id", 0)))
            tactical_score += cannon_count * 0.4
            
            # 远程兵有中等战术价值
            ranged_count = sum(1 for s in ally_soldiers if self._is_ranged_minion(s.get("config_id", 0)))
            tactical_score += ranged_count * 0.2
            
            # 总数量提供基础价值
            tactical_score += len(ally_soldiers) * 0.05
            
            wave_tactical_value = min(tactical_score, 1.0)
        
        # 7. 兵线节奏掌控
        wave_rhythm_mastery = 0.5
        
        # 评估对兵线节奏的掌控程度
        if my_hero and ally_soldiers:
            hero_pos = my_hero["actor_state"]["location"]
            
            # 计算英雄与兵线的距离
            min_distance = float('inf')
            for soldier in ally_soldiers:
                soldier_pos = soldier.get("location", {})
                dist = self.cal_dist(hero_pos, soldier_pos)
                min_distance = min(min_distance, dist)
            
            # 距离越近，节奏掌控越好
            if min_distance != float('inf'):
                distance_factor = max(0, 1.0 - min_distance / 5000.0)
                wave_rhythm_mastery = distance_factor
        
        # 8. 兵线预期管理
        wave_expectation_management = 0.5
        
        # 基于兵线状态对战局走向的预期管理
        current_frame = frame_state.get("frame_no", 0)
        game_phase = min(current_frame / (10 * 60 * 60), 1.0)  # 10分钟游戏
        
        if ally_soldiers and enemy_soldiers:
            ally_strength = len(ally_soldiers) + sum(2 for s in ally_soldiers if self._is_cannon_minion(s.get("config_id", 0)))
            enemy_strength = len(enemy_soldiers) + sum(2 for s in enemy_soldiers if self._is_cannon_minion(s.get("config_id", 0)))
            
            strength_ratio = ally_strength / max(ally_strength + enemy_strength, 1)
            
            # 游戏后期，兵线预期管理更重要
            wave_expectation_management = strength_ratio * (0.5 + game_phase * 0.5)
        
        # 9. 兵线信息战
        wave_information_warfare = 0.5
        
        # 评估通过兵线获取和隐藏信息的价值
        if ally_soldiers:
            # 兵线可以提供视野和情报
            vision_value = 0.0
            
            # 兵线数量越多，覆盖范围越大
            vision_value += min(len(ally_soldiers) / 6.0, 0.5)
            
            # 远程兵提供更好的视野
            ranged_count = sum(1 for s in ally_soldiers if self._is_ranged_minion(s.get("config_id", 0)))
            vision_value += min(ranged_count / 3.0, 0.3)
            
            # 分散度影响信息收集效果
            if len(ally_soldiers) > 1:
                positions = [(s.get("location", {}).get("x", 0), s.get("location", {}).get("z", 0)) for s in ally_soldiers]
                spread = max(positions)[0] - min(positions)[0] + max(positions, key=lambda p: p[1])[1] - min(positions, key=lambda p: p[1])[1]
                spread_bonus = min(spread / 10000.0, 0.2)
                vision_value += spread_bonus
            
            wave_information_warfare = min(vision_value, 1.0)
        
        # 10. 兵线终局思维
        wave_endgame_thinking = 0.5
        
        # 基于当前兵线状态进行终局思考
        current_frame = frame_state.get("frame_no", 0)
        game_duration = current_frame / 60.0  # 秒
        
        if ally_soldiers:
            # 游戏越接近后期，每个兵线决策越关键
            time_pressure = min(game_duration / 600.0, 1.0)  # 10分钟后满值
            
            # 兵线质量影响终局价值
            ally_cannon = sum(1 for s in ally_soldiers if self._is_cannon_minion(s.get("config_id", 0)))
            cannon_factor = min(ally_cannon * 0.3, 0.6)
            
            wave_endgame_thinking = time_pressure * 0.7 + cannon_factor * 0.3
        
        return [
            wave_intimidation_effect,
            wave_deception_potential,
            wave_pressure_perception,
            wave_psychological_advantage,
            wave_manipulation_difficulty,
            wave_tactical_value,
            wave_rhythm_mastery,
            wave_expectation_management,
            wave_information_warfare,
            wave_endgame_thinking,
            0.5,  # 兵线心理压力（占位符）
            0.5,  # 兵线博弈层次（占位符）
            0.5,  # 兵线情绪影响（占位符）
            0.5,  # 兵线战术欺骗（占位符）
            0.5,  # 兵线战略威慑（占位符）
        ]





    def _pos(self, actor_like):
        """
        统一取 (x, z)：
        - 常规 npc/hero: obj['location'] = {'x','z'}
        - hero 有时包在 obj['actor_state']['location']
        - 子弹可能是 obj['pos'] = {'x','z'} 或 obj['location']
        """
        if not actor_like:
            return 0.0, 0.0

        # 1) 直接 location
        loc = actor_like.get("location")
        if not loc and isinstance(actor_like.get("actor_state"), dict):
            # 2) 英雄/带 actor_state 的情况
            loc = (actor_like["actor_state"] or {}).get("location")
        if not loc:
            # 3) 子弹等可能使用 pos 字段
            loc = actor_like.get("pos")

        if not isinstance(loc, dict):
            return 0.0, 0.0

        x = float(loc.get("x", 0.0))
        z = float(loc.get("z", 0.0))
        return x, z



