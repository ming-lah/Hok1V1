#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

- Keep original rewards and interface.
- Remove grass rewards (grass_engage / grass_in), even if present in weights.
- Merge image rewards:
  Dense: hp_point(=hero_hp_point), tower_hp_point, money(gold), ep_rate, exp
  Sparse: death, kill, last_hit
  Aliases are supported to avoid refactor in configs.
"""

import math
from agent_ppo.conf.conf import GameConfig


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        self.RANGE_NORM = 15000.0

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frameNo"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # ---------------- helpers ---------------- #
    @staticmethod
    def _pos(o):
        try:
            if not isinstance(o, dict):
                return 0.0, 0.0
            loc = o.get("location") or {}
            return float(loc.get("x", 0.0)), float(loc.get("z", 0.0))
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _hp_ratio(u):
        if not isinstance(u, dict):
            return 0.0
        hp = float(u.get("hp", 0.0))
        mx = float(u.get("max_hp", 0.0))
        return hp / mx if mx > 0 else 0.0

    @staticmethod
    def _ep_ratio(actor_state):
        vals = (actor_state or {}).get("values", {}) or {}
        ep = float(vals.get("ep", 0.0))
        mx = float(vals.get("max_ep", 0.0))
        return ep / mx if mx > 0 else 0.0

    # ---------------- per-side frame calc ---------------- #
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        # towers
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":
                    main_spring = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":
                    enemy_spring = organ

        enemy_camp = (enemy_hero.get("actor_state") or {}).get("camp") if enemy_hero else None

        A = [n for n in npc_list if n.get("sub_type") == "ACTOR_SUB_SOLDIER" and n.get("camp") == camp and n.get("hp", 0) > 0]
        E = [n for n in npc_list if n.get("sub_type") == "ACTOR_SUB_SOLDIER" and n.get("camp") == enemy_camp and n.get("hp", 0) > 0]

        def _front_to_tower(lst, tower):
            if not lst or not tower:
                return self.RANGE_NORM
            tx, tz = self._pos(tower)
            best = self.RANGE_NORM
            for u in lst:
                ux, uz = self._pos(u)
                d = math.hypot(ux - tx, uz - tz)
                if d < best:
                    best = d
            return min(best, self.RANGE_NORM)

        a_front = _front_to_tower(A, enemy_tower)
        e_front = _front_to_tower(E, main_tower)
        push_depth = (e_front - a_front) / self.RANGE_NORM  # [-1,1]

        # tower danger & dive
        tower_danger = 0.0
        dive_no_minion = 0.0
        if main_hero and enemy_tower:
            me = (main_hero.get("actor_state") or {})
            mx, mz = self._pos(me)
            ex, ez = self._pos(enemy_tower)
            atk_r = float(enemy_tower.get("attack_range", 0.0))
            in_range = 1.0 if (atk_r > 0 and math.hypot(mx - ex, mz - ez) <= atk_r) else 0.0
            target_me = 1.0 if str(enemy_tower.get("attack_target", "")) == str(me.get("runtime_id", "")) else 0.0
            tower_danger = 1.0 if (in_range or target_me) else 0.0

            near_cnt = 0
            for u in A:
                ux, uz = self._pos(u)
                if atk_r > 0 and math.hypot(ux - ex, uz - ez) <= atk_r * 0.9:
                    near_cnt += 1
            dive_no_minion = 1.0 if (in_range and near_cnt == 0) else 0.0

        # ---- events: kill / death / last_hit ----
        kill_event, death_event, last_hit_event = 0.0, 0.0, 0.0
        acts = frame_data.get("frame_action", []) or []
        my_id = str((main_hero.get("actor_state") or {}).get("runtime_id", "")) if main_hero else ""
        if isinstance(acts, list):
            for a in acts:
                if not isinstance(a, dict):
                    continue
                da = a.get("dead_action") or {}
                if not isinstance(da, dict):
                    continue
                death = da.get("death") or {}
                killer = da.get("killer") or {}
                death_camp = death.get("camp", None)
                killer_camp = killer.get("camp", None)

                # hero-vs-hero
                if death.get("sub_type") in ("ACTOR_SUB_hero", "ACTOR_SUB_HERO", "hero"):
                    if killer_camp == camp:  # 我方击杀
                        kill_event += 1.0
                    if death_camp == camp:   # 我方阵亡
                        death_event += 1.0

                # last hit soldier
                if death.get("sub_type") in ("ACTOR_SUB_SOLDIER", "soldier", "SOLDIER"):
                    killer_id = str(killer.get("runtime_id", killer.get("config_id", "")))
                    if killer_id and killer_id == my_id:
                        last_hit_event += 1.0
                    elif killer_camp == camp:
                        last_hit_event += 1.0

        # ---- dense stats (my side) ----
        hp_ratio = self._hp_ratio((main_hero or {}).get("actor_state") or {})
        tower_ratio = self._hp_ratio(main_tower or {})
        gold_total = float(main_hero.get("money", 0.0) if main_hero else 0.0)
        ep_ratio = self._ep_ratio((main_hero or {}).get("actor_state") or {})
        exp_total = float(main_hero.get("exp", 0.0) if main_hero else 0.0)

        # ---- write per-key values (alias-friendly) ----
        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            # Dense (aliases)
            if reward_name in ("tower_hp_point",):
                reward_struct.cur_frame_value = tower_ratio
            elif reward_name in ("forward",):
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name in ("hero_hp_point", "hp_point"):
                reward_struct.cur_frame_value = hp_ratio
            elif reward_name in ("gold_point", "money", "gold", "money_point"):
                reward_struct.cur_frame_value = gold_total
            elif reward_name in ("minion_push_depth",):
                reward_struct.cur_frame_value = push_depth
            elif reward_name in ("ep_rate",):
                reward_struct.cur_frame_value = ep_ratio
            elif reward_name in ("exp", "exp_point"):
                reward_struct.cur_frame_value = exp_total

            # Events (aliases)
            elif reward_name in ("kill_event", "kill"):
                reward_struct.cur_frame_value = kill_event
            elif reward_name in ("death_event", "death"):
                reward_struct.cur_frame_value = death_event
            elif reward_name in ("last_hit_event", "last_hit"):
                reward_struct.cur_frame_value = last_hit_event

            # Grass rewards — removed/disabled explicitly
            elif reward_name in ("grass_engage", "grass_in", "grass"):
                reward_struct.cur_frame_value = 0.0

            else:
                reward_struct.cur_frame_value = 0.0

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"]) if main_tower else (0.0, 0.0)
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"]) if enemy_tower else (0.0, 0.0)
        hero_pos = (
            (main_hero or {}).get("actor_state", {}).get("location", {}).get("x", 0.0),
            (main_hero or {}).get("actor_state", {}).get("location", {}).get("z", 0.0),
        )
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos) if enemy_tower else 0.0
        dist_main2emy = max(math.dist(main_tower_pos, enemy_tower_pos), 1e-6) if main_tower and enemy_tower else 1.0
        base = (dist_main2emy - dist_hero2emy) / dist_main2emy
        base = max(0.0, base)  # 远离敌塔不奖励
        hp = float((main_hero or {}).get("actor_state", {}).get("hp", 0.0))
        mx = max(float((main_hero or {}).get("actor_state", {}).get("max_hp", 1.0)), 1.0)
        hp_scale = hp / mx
        return base * hp_scale

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0

        # absolute terms per-frame (not zero-sum)
        ABS_NAMES = {"forward", "tower_danger", "dive_no_minion"}
        # instantaneous event diff
        EVENT_NAMES = {"kill_event", "kill", "death_event", "death", "last_hit_event", "last_hit"}
        # removed grass rewards if they still appear in weights
        GRASS_NAMES = {"grass_engage", "grass_in", "grass"}

        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            w = reward_struct.weight or 0.0
            if w == 0.0 or reward_name in GRASS_NAMES:
                reward_struct.value = 0.0
                reward_dict[reward_name] = 0.0
                continue

            if reward_name in ABS_NAMES:
                cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.value = cur

            elif reward_name in EVENT_NAMES:
                cur_diff = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.value = cur_diff

            else:
                cur_diff = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                last_diff = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = cur_diff - last_diff

            weight_sum += w
            reward_sum += reward_struct.value * w
            reward_dict[reward_name] = reward_struct.value

        reward_dict["reward_sum"] = reward_sum
