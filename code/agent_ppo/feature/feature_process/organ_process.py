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
        self.one_unit_feature_num = 7
        self.unit_buff_num = 1

        # 其他特征属性
        self.MAP_NORM = 30000.0
        self.RANGE_NORM = 15000.0
        self.MINION_TOPK = 4
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



