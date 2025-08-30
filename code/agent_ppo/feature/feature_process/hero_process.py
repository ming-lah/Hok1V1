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


class HeroProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 3
        self.unit_buff_num = 1

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, frame_state):

        self.generate_hero_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")
        enemy_vec = self.generate_one_type_hero_feature(self.enemy_camp_hero_dict, "enemy_camp")
        return main_camp_hero_vector_feature + enemy_vec

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["actor_state"]["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["actor_state"]["config_id"]] = hero

    def generate_hero_info_dict(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()

        # Find our heroes and number them in order
        # 找到我方英雄并按照顺序编号
        for hero in frame_state["npc_states"]:
            if hero["sub_type"] != "ACTOR_SUB_hero" or hero["hp"] <= 0:
                continue
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["runtime_id"]] = hero
        self.main_camp_hero_dict = OrderedDict(sorted(self.main_camp_hero_dict.items()))

        # Find enemy heroes and number them in order
        # 找到敌方英雄并按照顺序编号
        for hero in frame_state["npc_states"]:
            if hero["sub_type"] != "ACTOR_SUB_hero" or hero["hp"] <= 0:
                continue
            if hero["camp"] != self.main_camp:
                self.enemy_camp_hero_dict[hero["runtime_id"]] = hero
        self.enemy_camp_hero_dict = OrderedDict(sorted(self.enemy_camp_hero_dict.items()))

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    value_vec = []
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["actor_state"]["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)



    # 新加入特征
    def get_forward_x(self, hero, out, feature_name):
        fx = hero.get("actor_state", {}).get("forward", {}).get("x", 0.0)
        fz = hero.get("actor_state", {}).get("forward", {}).get("z", 0.0)
        denom = math.sqrt(fx * fx + fz * fz) + 1e-8
        val = fx / denom
        if self.transform_camp2_to_camp1:
            val = -val
        out.append(float(val))  # ∈[-1,1]，INI 用 min_max:-1:1

    def get_forward_z(self, hero, out, feature_name):
        fx = hero.get("actor_state", {}).get("forward", {}).get("x", 0.0)
        fz = hero.get("actor_state", {}).get("forward", {}).get("z", 0.0)
        denom = math.sqrt(fx * fx + fz * fz) + 1e-8
        val = fz / denom
        if self.transform_camp2_to_camp1:
            val = -val
        out.append(float(val))  # ∈[-1,1]

    # ========== 比例/标量 ==========
    def get_hp_rate(self, hero, out, feature_name):
        st = hero.get("actor_state", {})
        hp = float(st.get("hp", 0.0))
        max_hp = float(st.get("max_hp", 1.0))
        out.append(0.0 if max_hp <= 0 else hp / max_hp)

    def get_ep_rate(self, hero, out, feature_name):
        vals = hero.get("actor_state", {}).get("values", {})
        ep = float(vals.get("ep", 0.0))
        max_ep = float(vals.get("max_ep", 1.0))
        out.append(0.0 if max_ep <= 0 else ep / max_ep)

    def get_level(self, hero, out, feature_name):
        out.append(float(hero.get("level", 0)))

    def get_money(self, hero, out, feature_name):
        out.append(float(hero.get("money", 0)))

    def get_attack_range(self, hero, out, feature_name):
        out.append(float(hero.get("actor_state", {}).get("attack_range", 0)))

    # ========== 核心属性（ActorValue） ==========
    def _get_val(self, hero, key, default=0.0):
        return float(hero.get("actor_state", {}).get("values", {}).get(key, default))

    def get_phy_atk(self, hero, out, feature_name): out.append(self._get_val(hero, "phy_atk"))
    def get_phy_def(self, hero, out, feature_name): out.append(self._get_val(hero, "phy_def"))
    def get_mgc_atk(self, hero, out, feature_name): out.append(self._get_val(hero, "mgc_atk"))
    def get_mgc_def(self, hero, out, feature_name): out.append(self._get_val(hero, "mgc_def"))
    def get_mov_spd(self, hero, out, feature_name): out.append(self._get_val(hero, "mov_spd"))
    def get_atk_spd(self, hero, out, feature_name): out.append(self._get_val(hero, "atk_spd"))
    def get_crit_rate(self, hero, out, feature_name): out.append(self._get_val(hero, "crit_rate"))
    def get_crit_effe(self, hero, out, feature_name): out.append(self._get_val(hero, "crit_effe"))
    def get_phy_armor_hurt(self, hero, out, feature_name): out.append(self._get_val(hero, "phy_armor_hurt"))
    def get_mgc_armor_hurt(self, hero, out, feature_name): out.append(self._get_val(hero, "mgc_armor_hurt"))
    def get_phy_vamp(self, hero, out, feature_name): out.append(self._get_val(hero, "phy_vamp"))
    def get_mgc_vamp(self, hero, out, feature_name): out.append(self._get_val(hero, "mgc_vamp"))
    def get_cd_reduce(self, hero, out, feature_name): out.append(self._get_val(hero, "cd_reduce"))
    def get_ctrl_reduce(self, hero, out, feature_name): out.append(self._get_val(hero, "ctrl_reduce"))

    # ========== 技能：前 4 个槽（含召唤师技能） ==========
    def _get_slot(self, hero, idx):
        slots = hero.get("skill_state", {}).get("slot_states", [])
        if 0 <= idx < len(slots):
            return slots[idx]
        return None

    def get_skill_cd_rate(self, hero, out, feature_name):
        # 特征名形如：skill0_cd_rate / skill1_cd_rate / skill2_cd_rate / skill3_cd_rate
        # 从特征名尾部解析索引
        idx = int(''.join([c for c in feature_name if c.isdigit()]) or 0)
        slot = self._get_slot(hero, idx)
        if not slot:
            out.append(0.0); return
        cd = float(slot.get("cooldown", 0.0))
        cd_max = float(slot.get("cooldown_max", 0.0))
        rate = 0.0 if cd_max <= 0 else (cd / cd_max)
        out.append(rate)  # 0~1

    def get_skill_usable(self, hero, out, feature_name):
        idx = int(''.join([c for c in feature_name if c.isdigit()]) or 0)
        slot = self._get_slot(hero, idx)
        usable = slot.get("usable", False) if slot else False
        out.append(1.0 if usable else 0.0)

    # ========== 其它语义 ==========
    def get_is_in_grass(self, hero, out, feature_name):
        out.append(1.0 if hero.get("isInGrass", False) else 0.0)

    def get_kill_cnt(self, hero, out, feature_name):
        out.append(float(hero.get("killCnt", 0)))

    def get_dead_cnt(self, hero, out, feature_name):
        out.append(float(hero.get("deadCnt", 0)))
    


    # ===== 新增：对抗派生特征（仅英雄信息） =====
    def _first_hero(self, hero_dict):
        # 1v1：取字典里第一名英雄即可
        return next(iter(hero_dict.values()), None)

    def _pos_xz(self, actor_or_state: dict):
        st = actor_or_state.get("actor_state", actor_or_state) or {}
        loc = st.get("location", {}) or {}
        return float(loc.get("x", 0.0)), float(loc.get("z", 0.0))

    def _hp_rate_of(self, hero: dict) -> float:
        st = hero.get("actor_state", {}) or {}
        hp = float(st.get("hp", 0.0))
        mx = float(st.get("max_hp", 1.0))
        return 0.0 if mx <= 0 else (hp / mx)

    def _atk_range_of(self, hero: dict) -> float:
        st = hero.get("actor_state", {}) or {}
        return float(st.get("attack_range", 0.0))

    def get_dist_to_enemy(self, hero, out, feature_name):
        # 依据 hero 阵营找到对面英雄，计算欧氏距离
        my_camp = (hero.get("actor_state") or {}).get("camp")
        if my_camp == self.main_camp:
            opp = self._first_hero(self.enemy_camp_hero_dict)
        else:
            opp = self._first_hero(self.main_camp_hero_dict)
        if not opp:
            out.append(0.0); return
        x1, z1 = self._pos_xz(hero)
        x2, z2 = self._pos_xz(opp)
        dist = math.hypot(x1 - x2, z1 - z2)  # 单位与坐标一致
        out.append(float(dist))              # 归一化由 ini 的 min_max 完成

    def get_in_my_atk_range(self, hero, out, feature_name):
        my_camp = (hero.get("actor_state") or {}).get("camp")
        if my_camp == self.main_camp:
            opp = self._first_hero(self.enemy_camp_hero_dict)
        else:
            opp = self._first_hero(self.main_camp_hero_dict)
        if not opp:
            out.append(0.0); return
        x1, z1 = self._pos_xz(hero)
        x2, z2 = self._pos_xz(opp)
        dist = math.hypot(x1 - x2, z1 - z2)
        in_range = 1.0 if dist <= self._atk_range_of(hero) else 0.0
        out.append(in_range)

    def get_in_enemy_atk_range(self, hero, out, feature_name):
        my_camp = (hero.get("actor_state") or {}).get("camp")
        if my_camp == self.main_camp:
            opp = self._first_hero(self.enemy_camp_hero_dict)
        else:
            opp = self._first_hero(self.main_camp_hero_dict)
        if not opp:
            out.append(0.0); return
        x1, z1 = self._pos_xz(hero)
        x2, z2 = self._pos_xz(opp)
        dist = math.hypot(x1 - x2, z1 - z2)
        in_enemy = 1.0 if dist <= self._atk_range_of(opp) else 0.0
        out.append(in_enemy)

    def get_hp_rate_diff(self, hero, out, feature_name):
        my_camp = (hero.get("actor_state") or {}).get("camp")
        if my_camp == self.main_camp:
            opp = self._first_hero(self.enemy_camp_hero_dict)
        else:
            opp = self._first_hero(self.main_camp_hero_dict)
        if not opp:
            out.append(0.0); return
        val = self._hp_rate_of(hero) - self._hp_rate_of(opp)  # ∈[-1,1]
        out.append(float(val))


    # ===== 新增：对抗增强特征（仅英雄字段） =====
    def _first_hero(self, hero_dict):
        return next(iter(hero_dict.values()), None)

    def _pos_xz(self, actor_or_state: dict):
        st = actor_or_state.get("actor_state", actor_or_state) or {}
        loc = st.get("location", {}) or {}
        return float(loc.get("x", 0.0)), float(loc.get("z", 0.0))

    def _camp_index(self, camp_str: str) -> int:
        # "PLAYERCAMP_1" -> 0,  "PLAYERCAMP_2" -> 1，兜底返回 0
        if isinstance(camp_str, str) and camp_str.endswith("_2"):
            return 1
        return 0

    def _hp_rate_of(self, hero: dict) -> float:
        st = hero.get("actor_state", {}) or {}
        hp = float(st.get("hp", 0.0)); mx = float(st.get("max_hp", 1.0))
        return 0.0 if mx <= 0 else (hp / mx)

    def _atk_range_of(self, hero: dict) -> float:
        st = hero.get("actor_state", {}) or {}
        return float(st.get("attack_range", 0.0))

    def _forward_xz(self, hero: dict):
        st = hero.get("actor_state", {}) or {}
        fx = float((st.get("forward") or {}).get("x", 0.0))
        fz = float((st.get("forward") or {}).get("z", 0.0))
        # 单位化并考虑阵营镜像（与 get_forward_x/z 一致）
        denom = math.sqrt(fx * fx + fz * fz) + 1e-8
        fx /= denom; fz /= denom
        if self.transform_camp2_to_camp1:
            fx = -fx; fz = -fz
        return fx, fz

    def get_rel_dir_cos(self, hero, out, feature_name):
        # 英雄面朝方向 与 指向敌人的向量 的余弦 ∈[-1,1]
        my_camp = (hero.get("actor_state") or {}).get("camp")
        opp = self._first_hero(self.enemy_camp_hero_dict if my_camp == self.main_camp else self.main_camp_hero_dict)
        if not opp:
            out.append(0.0); return
        x1, z1 = self._pos_xz(hero); x2, z2 = self._pos_xz(opp)
        dx, dz = (x2 - x1), (z2 - z1)
        # 阵营镜像，与位置特征一致
        if self.transform_camp2_to_camp1:
            dx = -dx; dz = -dz
        dist = math.hypot(dx, dz) + 1e-8
        vx, vz = dx / dist, dz / dist
        fx, fz = self._forward_xz(hero)
        cos_val = fx * vx + fz * vz
        out.append(float(max(-1.0, min(1.0, cos_val))))

    def _get_slot(self, hero, idx):
        slots = hero.get("skill_state", {}).get("slot_states", [])
        if 0 <= idx < len(slots):
            return slots[idx]
        return None

    def get_skill_usedTimes(self, hero, out, feature_name):
        idx = int(''.join([c for c in feature_name if c.isdigit()]) or 0)
        slot = self._get_slot(hero, idx)
        used = int((slot or {}).get("usedTimes", 0))
        out.append(float(used))

    def get_skill_hit_rate(self, hero, out, feature_name):
        idx = int(''.join([c for c in feature_name if c.isdigit()]) or 0)
        slot = self._get_slot(hero, idx) or {}
        used = int(slot.get("usedTimes", 0))
        hit  = int(slot.get("hitHeroTimes", 0))
        rate = 0.0 if used <= 0 else (hit / max(1, used))
        out.append(float(max(0.0, min(1.0, rate))))

    def get_hp_recover(self, hero, out, feature_name):
        vals = (hero.get("actor_state") or {}).get("values", {}) or {}
        out.append(float(vals.get("hp_recover", 0.0)))

    def get_ep_recover(self, hero, out, feature_name):
        vals = (hero.get("actor_state") or {}).get("values", {}) or {}
        out.append(float(vals.get("ep_recover", 0.0)))

    def get_visible_to_main_camp(self, hero, out, feature_name):
        st = hero.get("actor_state", {}) or {}
        visible = list(st.get("camp_visible", []) or [True, True])
        idx = self._camp_index(self.main_camp)
        out.append(1.0 if (idx < len(visible) and bool(visible[idx])) else 0.0)

    def get_attack_target_is_enemy(self, hero, out, feature_name):
        my_camp = (hero.get("actor_state") or {}).get("camp")
        opp = self._first_hero(self.enemy_camp_hero_dict if my_camp == self.main_camp else self.main_camp_hero_dict)
        if not opp:
            out.append(0.0); return
        st  = hero.get("actor_state", {}) or {}
        tgt = int(st.get("attack_target", -1))
        # runtime_id 更稳；若无则回退到 config_id
        opp_id = int((opp.get("actor_state") or {}).get("runtime_id", (opp.get("actor_state") or {}).get("config_id", -9999)))
        out.append(1.0 if tgt == opp_id and opp_id != -9999 else 0.0)
