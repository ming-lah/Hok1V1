#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""



import math
from agent_ppo.conf.conf import GameConfig


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
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

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
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

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero

        # Get both defense towers
        # 获取双方防御塔
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
    # 工具函数
    def _pos(o):
        try:
            if not isinstance(o, dict):
                return 0.0, 0.0
            loc = o.get("location")
            x = loc.get("x", 0.0), 0.0
            z = loc.get("z", 0.0), 0.0
            return x, z
        except Exception:
            return 0.0, 0.0

        
        def _hp_ratio(u):
            if not isinstance(u, dict):
                return 0.0
            hp = float(u.get("hp", 0.0))
            mx = float(u.get("max_hp", 0.0))
            return hp / mx if mx > 0 else 0.0

        A = [n for n in npc_list if n.get("sub_type") == "ACTOR_SUB_SOLDIER" and n.get("camp") == camp and n.get("hp", 0) > 0]
        E = [n for n in npc_list if n.get("sub_type") == "ACTOR_SUB_SOLDIER" and n.get("camp") == enemy_camp and n.get("hp", 0) > 0]
        
        def _front_to_tower(lst, tower):
            if not lst or not tower:
                return self.RANGE_NORM
            tx, tz = _pos(tower)
            best = self.RANGE_NORM
            for u in lst:
                ux, uz = _pos(u)
                d = math.hypot(ux - tx, uz - tz)
                if d < best:
                    best = d
            return min(best, self.RANGE_NORM)

        a_front = _front_to_tower(A, enemy_tower)
        e_front = _front_to_tower(E, main_tower)
        push_depth = (e_front - a_front) / self.RANGE_NORM  # [-1,1]

        tower_danger = 0.0
        dive_no_minion = 0.0
        if main_hero and enemy_tower:
            me = (main_hero.get("actor_state") or {})
            mx, mz = _pos(me)
            ex, ez = _pos(enemy_tower)
            atk_r = float(enemy_tower.get("attack_range", 0.0))
            in_range = 1.0 if (atk_r > 0 and math.hypot(mx - ex, mz - ez) <= atk_r) else 0.0
            target_me = 1.0 if str(enemy_tower.get("attack_target", "")) == str(me.get("runtime_id", "")) else 0.0
            tower_danger = 1.0 if (in_range or target_me) else 0.0

            near_cnt = 0
            for u in A:
                ux, uz = _pos(u)
                if atk_r > 0 and math.hypot(ux - ex, uz - ez) <= atk_r * 0.9:
                    near_cnt += 1
            dive_no_minion = 1.0 if (in_range and near_cnt == 0) else 0.0

        # --- 草丛埋伏（非零和） ---
        grass_engage = 0.0
        if main_hero and enemy_hero:
            a_in = 1 if (main_hero.get("isInGrass") or (main_hero.get("actor_state") or {}).get("isInGrass")) else 0
            e_in = 1 if (enemy_hero.get("isInGrass") or (enemy_hero.get("actor_state") or {}).get("isInGrass")) else 0
            if a_in and not e_in:
                ax, az = _pos((main_hero.get("actor_state") or {}))
                ex, ez = _pos((enemy_hero.get("actor_state") or {}))
                if math.hypot(ax - ex, az - ez) <= 5000.0:
                    grass_engage = 1.0

        # --- 事件：击杀/死亡（帧级） ---
        kill_event, death_event = 0.0, 0.0
        acts = frame_data.get("frame_action", []) or []
        if isinstance(acts, list):
            for a in acts:
                if not isinstance(a, dict):
                    continue
                da = a.get("dead_action") or {}
                if not isinstance(da, dict):
                    continue
                death = (da.get("death") or {}).get("camp", None)
                killer = (da.get("killer") or {}).get("camp", None)
                if killer == camp:
                    kill_event += 1.0
                if death == camp:
                    death_event += 1.0

        # --- 经济（我方） ---
        gold_point = 0.0
        if main_hero:
            astate = main_hero.get("actor_state") or {}
            gold_point = float(astate.get("gold", astate.get("money", 0.0)) or 0.0)

        # --- 英雄/塔血量比例 ---
        hero_hp_ratio = _hp_ratio((main_hero or {}).get("actor_state") or {})
        my_tower_hp_ratio = _hp_ratio(main_tower)


        
        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Tower health points
            # 塔血量
            if reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "hero_hp_point":
                reward_struct.cur_frame_value = hero_hp_ratio
            elif reward_name == "gold_point":
                reward_struct.cur_frame_value = gold_point
            elif reward_name == "minion_push_depth":
                reward_struct.cur_frame_value = push_depth
            elif reward_name == "kill_event":
                reward_struct.cur_frame_value = kill_event
            elif reward_name == "death_event":
                reward_struct.cur_frame_value = death_event
            elif reward_name == "tower_danger":
                reward_struct.cur_frame_value = tower_danger
            elif reward_name == "dive_no_minion":
                reward_struct.cur_frame_value = dive_no_minion
            elif reward_name == "grass_engage":
                reward_struct.cur_frame_value = grass_engage

            else:
                reward_struct.cur_frame_value = 0.0


    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
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

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0

        # 非零和即时值
        ABS_NAMES = {"forward", "tower_danger", "dive_no_minion", "grass_engage"}
        # 事件瞬时差（避免“下一帧反向跳变”）
        EVENT_NAMES = {"kill_event", "death_event"}

        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            w = reward_struct.weight
            if w == 0.0:
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
                reward_struct.value = cur_diff  # 仅使用瞬时差

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