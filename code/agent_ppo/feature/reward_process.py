#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
HoK 1v1 Reward (8 items) aligned with the doc table:
- dense : hp_point, tower_hp_point, money, ep_rate, exp
- sparse: death, kill, last_hit

Keep original structure & interfaces:
- RewardStruct / init_calc_frame_map / GameRewardManager
- Uses GameConfig.REWARD_WEIGHT_DICT & TIME_SCALE_ARG
"""
import math
from typing import Dict, Any, List, Tuple
import json
from agent_ppo.conf.conf import GameConfig


# --------------------------- data structs --------------------------- #
class RewardStruct:
    """Per-item accumulator with frame cache and weight."""
    def __init__(self, m_weight: float = 0.0):
        self.cur_frame_value: float = 0.0
        self.last_frame_value: float = 0.0
        self.value: float = 0.0
        self.weight: float = m_weight


def init_calc_frame_map() -> Dict[str, RewardStruct]:
    """Create a map[item_name] -> RewardStruct using GameConfig weights."""
    calc_frame_map: Dict[str, RewardStruct] = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


# ----------------------------- manager ----------------------------- #
class GameRewardManager:
    """
    Eight-item reward consistent with the doc:
      hp_point(=my hp_ratio), tower_hp_point(=my tower hp_ratio),
      money(=my total gold), ep_rate(=my mp_ratio), exp(=my exp),
      death(event: I die), kill(event: I kill enemy hero), last_hit(event: I last-hit a soldier)
    """
    DENSE_KEYS  = {"hp_point", "tower_hp_point", "money", "ep_rate", "exp"}
    SPARSE_KEYS = {"death", "kill", "last_hit"}

    def __init__(self, main_hero_runtime_id: int):
        self.main_hero_player_id = main_hero_runtime_id  # run-time id used to find my hero
        self.main_hero_camp = -1

        # three maps: "current frame (diff/delta)" containers for both sides + my output
        self.m_cur_calc_frame_map  = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map= init_calc_frame_map()

        self.time_scale_arg = GameConfig.TIME_SCALE_ARG

    # -------------------------- public API -------------------------- #
    def result(self, frame_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute each reward item this frame (weighted sum returned in dict["reward_sum"])."""
        self._frame_data_process(frame_data)
        reward_dict: Dict[str, float] = {}
        self._combine_reward(frame_data, reward_dict)

        # time-decay shaping if enabled
        frame_no = int(frame_data.get("frameNo", 0))
        if self.time_scale_arg and self.time_scale_arg > 0:
            decay = math.pow(0.6, 1.0 * frame_no / float(self.time_scale_arg))
            for k in reward_dict:
                reward_dict[k] *= decay
        return reward_dict

    # ----------------------- frame preprocessing --------------------- #
    def _get_heroes(self, frame_data: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Return (my_hero, enemy_hero) dicts."""
        me, enemy = None, None
        for h in frame_data.get("hero_states", []) or []:
            camp = (h.get("actor_state") or {}).get("camp", None)
            # my camp discovered earlier in _frame_data_process
            if camp == self.main_hero_camp:
                me = h
            else:
                enemy = h
        return me, enemy

    def _get_towers(self, frame_data: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Return (my_tower, enemy_tower)."""
        my_t, en_t = None, None
        for u in frame_data.get("npc_states", []) or []:
            if u.get("sub_type") != "ACTOR_SUB_TOWER":
                continue
            if u.get("camp") == self.main_hero_camp:
                my_t = u
            else:
                en_t = u
        return my_t, en_t

    @staticmethod
    def _hp_ratio(u: Dict) -> float:
        if not isinstance(u, dict):
            return 0.0
        hp = float(u.get("hp", 0.0))
        mx = float(u.get("max_hp", 0.0))
        return hp / mx if mx > 0 else 0.0

    @staticmethod
    def _actor_state(hero: Dict) -> Dict:
        return (hero or {}).get("actor_state") or {}

    @staticmethod
    def _get_gold(astate: Dict) -> float:
        # prefer explicit gold; fallback to money
        return float(astate.get("gold", astate.get("money", 0.0)) or 0.0)

    @staticmethod
    def _get_ep_rate(astate: Dict) -> float:
        ep  = float((astate.get("values") or {}).get("ep", 0.0))
        mx  = float((astate.get("values") or {}).get("max_ep", 1.0))
        return 0.0 if mx <= 0 else ep / mx

    @staticmethod
    def _get_exp(hero: Dict) -> float:
        try:
            return float(hero.get("exp", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _is_subtype(obj: Dict, names: Tuple[str, ...]) -> bool:
        """Check subtype tags robustly."""
        if not isinstance(obj, dict):
            return False
        for k in ("sub_type", "actor_sub_type", "actor_type", "type"):
            v = obj.get(k, None)
            if isinstance(v, str) and v in names:
                return True
        return False

    def _collect_events(self, frame_data):
        """
        Robustly parse kill/death/last_hit from frame_action.
        Returns (my_kill_hero, my_death, my_last_hit_soldier) for THIS FRAME.
        - frame_action may be None / dict / list / string (even JSON string).
        - non-dict entries are ignored safely.
        """
        def _ensure_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, dict):
                return [x]
            if isinstance(x, str):
                # try JSON -> dict/list; else treat as noise
                try:
                    y = json.loads(x)
                    return _ensure_list(y)
                except Exception:
                    return []
            return []

        def _get(d, *keys, default=None):
            cur = d if isinstance(d, dict) else {}
            for k in keys:
                if not isinstance(cur, dict):
                    return default
                cur = cur.get(k)
            return default if cur is None else cur

        def _is_subtype(obj, names):
            if not isinstance(obj, dict):
                return False
            for k in ("sub_type", "actor_sub_type", "actor_type", "type"):
                v = obj.get(k)
                if isinstance(v, str) and v in names:
                    return True
            return False

        acts = _ensure_list(frame_data.get("frame_action"))

        my_kill = my_death = my_last_hit = 0

        for a in acts:
            if not isinstance(a, dict):
                continue  # skip strings, numbers, etc.

            # 1) 优先解析 dead_action 结构
            da = a.get("dead_action") or a.get("deadAction")
            if isinstance(da, dict):
                killer = da.get("killer") or {}
                death  = da.get("death") or {}

                killer_camp = killer.get("camp", killer.get("player_camp"))
                death_camp  = death.get("camp",  death.get("player_camp"))

                if _is_subtype(death, ("ACTOR_SUB_HERO", "HERO")):
                    if killer_camp == self.main_hero_camp:
                        my_kill += 1
                    if death_camp == self.main_hero_camp:
                        my_death += 1

                if _is_subtype(death, ("ACTOR_SUB_SOLDIER", "SOLDIER", "MINION")):
                    if killer_camp == self.main_hero_camp:
                        my_last_hit += 1
                continue  # 本条解析完，处理下一条

            # 2) 兼容另一类扁平事件：type/camp/target 等
            #    这里只做最宽松的兜底，不影响 dead_action 的主流程
            etype = (a.get("type") or a.get("event_type") or "").lower()
            camp  = a.get("camp") or a.get("from_camp")
            tgt   = a.get("target") or a.get("death") or {}
            if isinstance(tgt, dict):
                tgt_is_hero    = _is_subtype(tgt, ("ACTOR_SUB_HERO", "HERO"))
                tgt_is_soldier = _is_subtype(tgt, ("ACTOR_SUB_SOLDIER", "SOLDIER", "MINION"))
            else:
                tgt_is_hero = tgt_is_soldier = False

            # 仅当有明确 camp 且等于我方时计入
            if camp == self.main_hero_camp:
                if etype in ("kill", "hero_kill") or tgt_is_hero:
                    my_kill += 1
                if etype in ("last_hit", "soldier_last_hit") or tgt_is_soldier:
                    my_last_hit += 1
            # 自身死亡有时以 'death' 标记在自己侧
            if etype in ("death", "hero_death") and camp == self.main_hero_camp:
                my_death += 1

        return my_kill, my_death, my_last_hit


    def _frame_data_process_one_side(self, calc_map: Dict[str, RewardStruct], frame_data: Dict[str, Any], for_my_side: bool):
        """Fill per-item current values for one side (my or enemy) in calc_map."""
        # set camp for "my side" once
        if for_my_side and self.main_hero_camp == -1:
            for h in frame_data.get("hero_states", []) or []:
                if h.get("player_id") == self.main_hero_player_id:
                    self.main_hero_camp = (h.get("actor_state") or {}).get("camp", -1)
                    break

        # identify heroes/towers from the perspective of "my side"
        my_hero, enemy_hero = self._get_heroes(frame_data)
        my_tower, enemy_tower = self._get_towers(frame_data)

        # dense items (current absolute values)
        my_astate = self._actor_state(my_hero)
        calc_vals = {
            "hp_point":        self._hp_ratio(my_astate),
            "tower_hp_point":  self._hp_ratio(my_tower or {}),
            "money":           self._get_gold(my_astate),
            "ep_rate":         self._get_ep_rate(my_astate),
            "exp":             self._get_exp(my_hero or {}),
        }

        # sparse items (events this frame)
        my_kill, my_death, my_last_hit = self._collect_events(frame_data)
        calc_vals.update({
            "kill":      float(my_kill),
            "death":     float(my_death),
            "last_hit":  float(my_last_hit),
        })

        # write into calc_map (advance last->cur)
        for name, rs in calc_map.items():
            rs.last_frame_value = rs.cur_frame_value
            rs.cur_frame_value  = float(calc_vals.get(name, 0.0))

    def _frame_data_process(self, frame_data: Dict[str, Any]):
        """Populate per-side maps for this frame."""
        self._frame_data_process_one_side(self.m_main_calc_frame_map, frame_data, for_my_side=True)

        # Build an "enemy-side" view: swap camps logically by flipping which hero/tower we read.
        # Here we simply reuse the same parsers but with camp flipped when combining (see _combine_reward).
        # For compatibility with the old pipeline, we still fill enemy map using the *enemy* hero/tower values.
        # (We do this by temporarily swapping self.main_hero_camp)
        orig_camp = self.main_hero_camp
        # try to guess enemy camp
        enemy_camp = None
        for h in frame_data.get("hero_states", []) or []:
            c = (h.get("actor_state") or {}).get("camp", None)
            if c is not None and c != orig_camp:
                enemy_camp = c
                break
        self.main_hero_camp = enemy_camp if enemy_camp is not None else -999
        self._frame_data_process_one_side(self.m_enemy_calc_frame_map, frame_data, for_my_side=False)
        self.main_hero_camp = orig_camp

    # ------------------------ reward combining ------------------------ #
    def _combine_reward(self, frame_data: Dict[str, Any], out_dict: Dict[str, float]):
        """Combine dense (delta of advantage) and sparse (my events) into final per-item values and sum."""
        out_dict.clear()
        reward_sum = 0.0

        # Ensure only configured items are processed
        keys = list(self.m_cur_calc_frame_map.keys())

        for name in keys:
            rs_out = self.m_cur_calc_frame_map[name]
            w = rs_out.weight or 0.0

            if name in self.DENSE_KEYS:
                # advantage change = (my - enemy)_t - (my - enemy)_{t-1}
                my_cur  = self.m_main_calc_frame_map[name].cur_frame_value
                en_cur  = self.m_enemy_calc_frame_map[name].cur_frame_value
                my_last = self.m_main_calc_frame_map[name].last_frame_value
                en_last = self.m_enemy_calc_frame_map[name].last_frame_value
                rs_out.value = (my_cur - en_cur) - (my_last - en_last)

            elif name in self.SPARSE_KEYS:
                # my event count fired in this frame (do not subtract enemy)
                my_cur  = self.m_main_calc_frame_map[name].cur_frame_value
                my_last = self.m_main_calc_frame_map[name].last_frame_value
                rs_out.value = my_cur - my_last  # typically 0 or 1

            else:
                # unknown item -> zero
                rs_out.value = 0.0

            out_dict[name] = rs_out.value
            reward_sum += rs_out.value * w

        out_dict["reward_sum"] = reward_sum
