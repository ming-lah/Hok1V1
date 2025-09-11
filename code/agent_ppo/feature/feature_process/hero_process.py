#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Hero public features (HoK 1v1) — aligned to AI Arena doc, indices 0..101 per hero.
Excludes private_hero_feature (starts at 102 in the doc).

Key points
- Exact field order follows the doc table (public_hero_feature 0..101).
- Multi-dim entries are expanded inside getters:
  * hero_level -> 15-d one-hot (levels 1..15)
  * good_skill_buff_on_hero_itself / avoid_skill_control / blood_return -> 4-d each
  * frd_1v1_cake -> 3-d (rel x, rel z, exists)
  * all_equipskill_state -> 11-d multi-hot
- Geometry is mirrored for PLAYERCAMP_2 to keep a canonical frame.
- Graceful defaults (missing keys -> 0) to keep length stable.

Output per call:
  [public_main_hero (102 dims)] + [public_enemy_hero (102 dims)]  => 204 dims total.
"""
import configparser
import math
import os
from typing import Dict, Any, List

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer


# ------------------------------ helpers ------------------------------ #
def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _i(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)

def _b(x) -> float:
    return 1.0 if bool(x) else 0.0

def _safe_get(d: Dict, *keys, default=None):
    cur = d or {}
    for k in keys:
        cur = cur.get(k, {} if k != keys[-1] else default)
        if cur is None:
            return default
    return cur


# ================================ Core ================================= #
class HeroProcess:
    def __init__(self, camp: str):
        """
        camp: 'PLAYERCAMP_1' or 'PLAYERCAMP_2'
        """
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.mirror = (camp == "PLAYERCAMP_2")

        self.feature_order: List[str] = []
        self.feature_func_map = {}
        self.norm_table = {}

        self._frame_state = None
        self._main_hero = None
        self._enemy_hero = None

        self.get_hero_config()

    # ---------------------------- config IO ---------------------------- #
    def get_hero_config(self):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        here = os.path.dirname(__file__)
        ini_path = os.path.join(here, "hero_feature_config.ini")
        if not os.path.exists(ini_path):
            raise FileNotFoundError(f"hero_feature_config.ini not found at {ini_path}")
        cfg.read(ini_path, encoding="utf-8")

        # normalization table
        lines = [f"{k}:{v}" for k, v in cfg["feature_config"].items()]
        self.norm_table = self.normalizer.parse_config(lines)

        # function order & binding
        self.feature_order = []
        self.feature_func_map = {}
        for fname, getter in cfg["feature_functions"].items():
            self.feature_order.append(fname)
            if not hasattr(self, getter):
                raise ValueError(f"Getter '{getter}' not found for feature '{fname}'")
            self.feature_func_map[fname] = getattr(self, getter)

    # ------------------------------ entry ------------------------------ #
    def process_vec_hero(self, frame_state: Dict[str, Any]) -> List[float]:
        self._frame_state = frame_state or {}
        self._select_heroes(self._frame_state)

        main_vec = self._vectorize_side(self._main_hero)
        enemy_vec = self._vectorize_side(self._enemy_hero)
        return main_vec + enemy_vec

    # ------------------------- hero selection -------------------------- #
    def _select_heroes(self, frame_state: Dict[str, Any]):
        self._main_hero = None
        self._enemy_hero = None
        for h in frame_state.get("hero_states", []) or []:
            camp = _safe_get(h, "actor_state", "camp", default=None)
            if camp == self.main_camp:
                self._main_hero = h
            else:
                self._enemy_hero = h

    # -------------------------- vectorization -------------------------- #
    def _vectorize_side(self, hero: Dict[str, Any]) -> List[float]:
        out: List[float] = []
        if hero is None:
            # pad zeros per-feature using INI (so multi-dim entries are emitted correctly)
            for fname in self.feature_order:
                getter = self.feature_func_map[fname]
                raw_vals: List[float] = []
                getter(None, raw_vals, fname)  # getters handle None -> zeros
                n = self.norm_table.get(fname, None)
                if n is None:
                    out.extend(raw_vals if raw_vals else [0.0])
                else:
                    fn, *ps = n
                    for rv in raw_vals if raw_vals else [0.0]:
                        nv = fn(rv, *ps)
                        out.extend(nv if isinstance(nv, list) else [nv])
            return out

        for fname in self.feature_order:
            getter = self.feature_func_map[fname]
            raw_vals: List[float] = []
            getter(hero, raw_vals, fname)  # fill raw_vals (may push N>1 values)
            n = self.norm_table.get(fname, None)
            if n is None:
                # fall back: append raw (already 0/1 or unscaled floats)
                out.extend(raw_vals if raw_vals else [0.0])
            else:
                fn, *ps = n
                for rv in raw_vals if raw_vals else [0.0]:
                    nv = fn(rv, *ps)
                    out.extend(nv if isinstance(nv, list) else [nv])
        return out

    # ========================== basic helpers ========================== #
    def _pos(self, hero):
        if hero is None:
            return 0.0, 0.0
        x = _f(_safe_get(hero, "actor_state", "location", default={}).get("x", 0.0))
        z = _f(_safe_get(hero, "actor_state", "location", default={}).get("z", 0.0))
        if self.mirror:
            x, z = -x, -z
        return x, z

    def _forward(self, hero):
        if hero is None:
            return 0.0, 0.0
        fx = _f(_safe_get(hero, "actor_state", "forward", default={}).get("x", 0.0))
        fz = _f(_safe_get(hero, "actor_state", "forward", default={}).get("z", 0.0))
        n = math.hypot(fx, fz) + 1e-8
        fx, fz = (fx / n, fz / n) if n > 0 else (0.0, 0.0)
        if self.mirror:
            fx, fz = -fx, -fz
        return fx, fz

    def _enemy(self, hero):
        if hero is self._main_hero:
            return self._enemy_hero
        return self._main_hero

    def _panel(self, hero, key, default=0.0):
        if hero is None:
            return 0.0
        return _f(_safe_get(hero, "actor_state", "values", default={}).get(key, default))

    # ======================= 0..101: public fields ===================== #
    # 0: is_hero_alive
    def is_hero_alive(self, hero, out, _name):
        hp = 0.0 if hero is None else _f(_safe_get(hero, "actor_state", "hp", default=0))
        out.append(1.0 if hp > 0.0 else 0.0)

    # 1..15: hero_level (one-hot over levels 1..15)
    def hero_level(self, hero, out, _name):
        lvl = 0 if hero is None else _i(hero.get("level", 0))
        for k in range(1, 16):
            out.append(1.0 if lvl == k else 0.0)

    # scalars 16..48 (hp .. revive_time) + 49: kill_income
    def hp(self, hero, out, _): out.append(0.0 if hero is None else _f(_safe_get(hero, "actor_state", "hp", default=0)))
    def hp_rate(self, hero, out, _):
        if hero is None: out.append(0.0); return
        hp = _f(_safe_get(hero, "actor_state", "hp", default=0))
        mx = _f(_safe_get(hero, "actor_state", "max_hp", default=1))
        out.append(0.0 if mx <= 0 else hp / mx)
    def max_hp(self, hero, out, _): out.append(0.0 if hero is None else _f(_safe_get(hero, "actor_state", "max_hp", default=1)))
    def hp_recover(self, hero, out, _): out.append(self._panel(hero, "hp_recover"))
    def ep(self, hero, out, _): out.append(0.0 if hero is None else _f(_safe_get(hero, "actor_state", "values", default={}).get("ep", 0)))
    def ep_rate(self, hero, out, _):
        if hero is None: out.append(0.0); return
        ep = _f(_safe_get(hero, "actor_state", "values", default={}).get("ep", 0))
        mx = _f(_safe_get(hero, "actor_state", "values", default={}).get("max_ep", 1))
        out.append(0.0 if mx <= 0 else ep / mx)
    def max_ep(self, hero, out, _): out.append(0.0 if hero is None else _f(_safe_get(hero, "actor_state", "values", default={}).get("max_ep", 1)))
    def ep_recover(self, hero, out, _): out.append(self._panel(hero, "ep_recover"))
    def phy_atk(self, hero, out, _): out.append(self._panel(hero, "phy_atk"))
    def mgc_atk(self, hero, out, _): out.append(self._panel(hero, "mgc_atk"))
    def phy_def(self, hero, out, _): out.append(self._panel(hero, "phy_def"))
    def mgc_def(self, hero, out, _): out.append(self._panel(hero, "mgc_def"))
    def kill_cnt(self, hero, out, _): out.append(0.0 if hero is None else _f(hero.get("killCnt", 0)))
    def dead_cnt(self, hero, out, _): out.append(0.0 if hero is None else _f(hero.get("deadCnt", 0)))
    def money_cnt(self, hero, out, _):  # doc separates money_cnt and money
        if hero is None: out.append(0.0); return
        # prefer explicit money_cnt; fallback to money
        out.append(_f(hero.get("money_cnt", hero.get("money", 0))))
    def location_x(self, hero, out, _): x, _ = self._pos(hero); out.append(x)
    def location_z(self, hero, out, _): _, z = self._pos(hero); out.append(z)
    def dist_from_all_heros(self, hero, out, _):
        if hero is None: out.append(0.0); return
        opp = self._enemy(hero)
        if not opp:
            out.append(0.0); return
        x1, z1 = self._pos(hero); x2, z2 = self._pos(opp)
        out.append(math.hypot(x1 - x2, z1 - z2))
    def hero_move_speed(self, hero, out, _): out.append(self._panel(hero, "mov_spd"))
    def hero_attack_range(self, hero, out, _): out.append(0.0 if hero is None else _f(_safe_get(hero, "actor_state", "attack_range", default=0)))
    def hero_attack_speed(self, hero, out, _): out.append(self._panel(hero, "atk_spd"))
    def phy_armor_hurt(self, hero, out, _): out.append(self._panel(hero, "phy_armor_hurt"))
    def mgc_armor_hurt(self, hero, out, _): out.append(self._panel(hero, "mgc_armor_hurt"))
    def crit_rate(self, hero, out, _): out.append(self._panel(hero, "crit_rate"))
    def crit_effe(self, hero, out, _): out.append(self._panel(hero, "crit_effe"))
    def phy_vamp(self, hero, out, _): out.append(self._panel(hero, "phy_vamp"))
    def mgc_vamp(self, hero, out, _): out.append(self._panel(hero, "mgc_vamp"))
    def cd_reduce(self, hero, out, _): out.append(self._panel(hero, "cd_reduce"))
    def ctrl_reduce(self, hero, out, _): out.append(self._panel(hero, "ctrl_reduce"))
    def exp(self, hero, out, _): out.append(0.0 if hero is None else _f(hero.get("exp", 0)))
    def money(self, hero, out, _): out.append(0.0 if hero is None else _f(hero.get("money", 0)))
    def revive_time(self, hero, out, _): out.append(0.0 if hero is None else _f(hero.get("revive_time", 0)))
    def kill_income(self, hero, out, _): out.append(0.0 if hero is None else _f(hero.get("kill_income", 0)))

    # 50..57: skill usable & CD (raw ms)
    def skill_1_useable(self, hero, out, _): out.append(0.0 if hero is None else _b(_safe_get(hero, "skill_state", "slot_states", default=[])[:1] and _safe_get(hero, "skill_state", "slot_states", default=[])[0].get("usable", False)))
    def hero_skill_1_cd(self, hero, out, _):
        if hero is None: out.append(0.0); return
        s = (_safe_get(hero, "skill_state", "slot_states", default=[]) or [])
        cd = _f(s[0].get("cooldown", 0.0)) if len(s) > 0 else 0.0
        out.append(cd)
    def skill_2_useable(self, hero, out, _): out.append(0.0 if hero is None else _b(_safe_get(hero, "skill_state", "slot_states", default=[]) and _safe_get(hero, "skill_state", "slot_states", default=[])[1].get("usable", False) if len(_safe_get(hero,"skill_state","slot_states",default=[]))>1 else False))
    def hero_skill_2_cd(self, hero, out, _):
        if hero is None: out.append(0.0); return
        s = (_safe_get(hero, "skill_state", "slot_states", default=[]) or [])
        cd = _f(s[1].get("cooldown", 0.0)) if len(s) > 1 else 0.0
        out.append(cd)
    def skill_3_useable(self, hero, out, _): out.append(0.0 if hero is None else _b(_safe_get(hero, "skill_state", "slot_states", default=[]) and _safe_get(hero, "skill_state", "slot_states", default=[])[2].get("usable", False) if len(_safe_get(hero,"skill_state","slot_states",default=[]))>2 else False))
    def hero_skill_3_cd(self, hero, out, _):
        if hero is None: out.append(0.0); return
        s = (_safe_get(hero, "skill_state", "slot_states", default=[]) or [])
        cd = _f(s[2].get("cooldown", 0.0)) if len(s) > 2 else 0.0
        out.append(cd)
    def skill_4_useable(self, hero, out, _): out.append(0.0 if hero is None else _b(_safe_get(hero, "skill_state", "slot_states", default=[]) and _safe_get(hero, "skill_state", "slot_states", default=[])[3].get("usable", False) if len(_safe_get(hero,"skill_state","slot_states",default=[]))>3 else False))
    def hero_skill_4_cd(self, hero, out, _):
        if hero is None: out.append(0.0); return
        s = (_safe_get(hero, "skill_state", "slot_states", default=[]) or [])
        cd = _f(s[3].get("cooldown", 0.0)) if len(s) > 3 else 0.0
        out.append(cd)

    # 58..59: skill3 effect
    def is_skill3_effect_on(self, hero, out, _):
        if hero is None: out.append(0.0); return
        # heuristic sources (fallback to False)
        v = _safe_get(hero, "skill_state", "skill3_effect_on", default=None)
        if v is None:
            v = _safe_get(hero, "actor_state", "skill3_effect_on", default=False)
        out.append(_b(v))
    def remaining_time_of_skill3_effect(self, hero, out, _):
        if hero is None: out.append(0.0); return
        v = _safe_get(hero, "skill_state", "skill3_effect_left_ms", default=None)
        if v is None:
            v = _safe_get(hero, "actor_state", "skill3_effect_left_ms", default=0.0)
        out.append(_f(v, 0.0))

    # 59..71: three 4-d groups (buffs)
    def good_skill_buff_on_hero_itself(self, hero, out, _):
        # 4 flags; if unavailable, zeros
        arr = _safe_get(hero, "buff_state", "good_skill_buff_on_self", default=[0,0,0,0]) or [0,0,0,0]
        arr = list(arr) + [0,0,0,0]
        for k in range(4): out.append(_b(arr[k]))
    def avoid_skill_control(self, hero, out, _):
        arr = _safe_get(hero, "buff_state", "avoid_skill_control", default=[0,0,0,0]) or [0,0,0,0]
        arr = list(arr) + [0,0,0,0]
        for k in range(4): out.append(_b(arr[k]))
    def blood_return(self, hero, out, _):
        arr = _safe_get(hero, "buff_state", "blood_return", default=[0,0,0,0]) or [0,0,0,0]
        arr = list(arr) + [0,0,0,0]
        for k in range(4): out.append(_b(arr[k]))

    # 71..73: heal & summon skill CD
    def heal_skill_cd(self, hero, out, _):
        if hero is None: out.append(0.0); return
        cd = _safe_get(hero, "summon_state", "heal_cd", default=None)
        if cd is None:
            cd = _safe_get(hero, "actor_state", "heal_cd", default=0.0)
        out.append(_f(cd, 0.0))
    def summon_skill_cd(self, hero, out, _):
        if hero is None: out.append(0.0); return
        cd = _safe_get(hero, "summon_state", "summon_cd", default=None)
        if cd is None:
            cd = _safe_get(hero, "actor_state", "summon_cd", default=0.0)
        out.append(_f(cd, 0.0))

    # 73..83: 10 flags for summoner skill types
    _SUMMONER_TYPES = ["heal","sprint","punish","execute","rage","disrupt","daze","purify","weak","flash"]
    def _summon_flags(self, hero):
        flags = [0.0]*10
        if hero is None:
            return flags
        # try discrete type/code; support list (rare) or string code
        tp = _safe_get(hero, "summon_state", "type", default=None)
        if tp is None:
            tp = _safe_get(hero, "actor_state", "summon_type", default=None)
        if isinstance(tp, (list, tuple)):
            for t in tp:
                if isinstance(t, str) and t in self._SUMMONER_TYPES:
                    flags[self._SUMMONER_TYPES.index(t)] = 1.0
        elif isinstance(tp, str):
            if tp in self._SUMMONER_TYPES:
                flags[self._SUMMONER_TYPES.index(tp)] = 1.0
        elif isinstance(tp, int):
            # optional numeric code mapping (0..9)
            if 0 <= tp < 10:
                flags[tp] = 1.0
        # also allow explicit booleans (summon_skill_heal=1, etc.)
        for idx, name in enumerate(self._SUMMONER_TYPES):
            v = _safe_get(hero, "summon_state", f"{name}", default=None)
            if v is None:
                v = _safe_get(hero, "actor_state", f"summon_skill_{name}", default=None)
            if v is not None:
                flags[idx] = 1.0 if bool(v) else flags[idx]
        return flags

    def summon_skill_heal(self, hero, out, _):   out.append(self._summon_flags(hero)[0])
    def summon_skill_sprint(self, hero, out, _): out.append(self._summon_flags(hero)[1])
    def summon_skill_punish(self, hero, out, _): out.append(self._summon_flags(hero)[2])
    def summon_skill_execute(self, hero, out, _):out.append(self._summon_flags(hero)[3])
    def summon_skill_rage(self, hero, out, _):   out.append(self._summon_flags(hero)[4])
    def summon_skill_disrupt(self, hero, out, _):out.append(self._summon_flags(hero)[5])
    def summon_skill_daze(self, hero, out, _):   out.append(self._summon_flags(hero)[6])
    def summon_skill_purify(self, hero, out, _): out.append(self._summon_flags(hero)[7])
    def summon_skill_weak(self, hero, out, _):   out.append(self._summon_flags(hero)[8])
    def summon_skill_flash(self, hero, out, _):  out.append(self._summon_flags(hero)[9])

    # 84: common attack usable
    def common_skill_is_useable(self, hero, out, _):
        if hero is None: out.append(0.0); return
        val = _safe_get(hero, "actor_state", "common_skill_is_useable", default=None)
        out.append(1.0 if bool(val) else 0.0)

    # 85..87: tower-range flags
    def hero_in_main_camp_tower_atk_range(self, hero, out, _):
        if hero is None: out.append(0.0); return
        v = _safe_get(hero, "actor_state", "in_main_tower_range", default=None)
        out.append(_b(v))
    def hero_in_enemy_camp_tower_atk_range(self, hero, out, _):
        if hero is None: out.append(0.0); return
        v = _safe_get(hero, "actor_state", "in_enemy_tower_range", default=None)
        out.append(_b(v))
    def is_hero_under_tower_atk(self, hero, out, _):
        if hero is None: out.append(0.0); return
        v = _safe_get(hero, "actor_state", "under_tower_attack", default=None)
        out.append(_b(v))

    # 87..90: frd_1v1_cake: (rel x, rel z, exists)
    def frd_1v1_cake(self, hero, out, _):
        if hero is None:
            out.extend([0.0, 0.0, 0.0]); return
        hx, hz = self._pos(hero)
        # try several plausible locations in frame_state
        cake = _safe_get(self._frame_state, "global_state", "cake", default=None)
        if cake is None:
            cake = _safe_get(self._frame_state, "vec_feature_global", "cake", default=None)
        if cake is None:
            cake = _safe_get(self._frame_state, "cake_state", default=None)
        if isinstance(cake, dict):
            cx = _f(_safe_get(cake, "location", default={}).get("x", 0.0))
            cz = _f(_safe_get(cake, "location", default={}).get("z", 0.0))
            if self.mirror:
                cx, cz = -cx, -cz
            out.extend([cx - hx, cz - hz, 1.0])
        else:
            out.extend([0.0, 0.0, 0.0])

    # 90..101: all_equipskill_state (11-d multi-hot), available_equipskill_cd
    _EQUIP_ACTIVE_SLOTS = 11
    def all_equipskill_state(self, hero, out, _):
        if hero is None:
            out.extend([0.0]*self._EQUIP_ACTIVE_SLOTS); return
        states = _safe_get(hero, "equip_skill_state", "active_states", default=None)
        if states is None:
            # allow boolean fields like equip_active_0 .. equip_active_10
            states = [ _b(_safe_get(hero, "equip_skill_state", f"equip_active_{i}", default=0)) for i in range(self._EQUIP_ACTIVE_SLOTS) ]
        else:
            states = list(states)
        states += [0.0]*self._EQUIP_ACTIVE_SLOTS
        for i in range(self._EQUIP_ACTIVE_SLOTS):
            out.append(_b(states[i]))
    def available_equipskill_cd(self, hero, out, _):
        if hero is None: out.append(0.0); return
        cd = _safe_get(hero, "equip_skill_state", "available_cd", default=None)
        if cd is None:
            cd = _safe_get(hero, "actor_state", "equip_available_cd", default=0.0)
        out.append(_f(cd, 0.0))
