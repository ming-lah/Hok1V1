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

    # =================== ENHANCED FEATURES =================== #
    # 102..113: Skill Prediction & Combo Features (12 dims)
    def skill_combo_ready(self, hero, out, _):
        """技能连击就绪状态"""
        if hero is None: out.append(0.0); return
        
        # 检查所有技能是否都可用（连击就绪）
        s = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        all_ready = True
        for i in range(min(3, len(s))):  # 检查前3个技能
            if not s[i].get("usable", False):
                all_ready = False
                break
        out.append(1.0 if all_ready else 0.0)

    def skill_1_enemy_in_range(self, hero, out, _):
        """技能1范围内是否有敌人"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 获取技能1范围（估算）
        skill_range = _f(_safe_get(hero, "skill_state", "skill_1_range", default=5000))
        if skill_range <= 0: skill_range = 5000  # 默认范围
        
        # 计算距离
        hx, hz = self._pos(hero); ex, ez = self._pos(enemy)
        dist = math.hypot(hx - ex, hz - ez)
        out.append(1.0 if dist <= skill_range else 0.0)

    def skill_2_enemy_in_range(self, hero, out, _):
        """技能2范围内是否有敌人"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        skill_range = _f(_safe_get(hero, "skill_state", "skill_2_range", default=6000))
        if skill_range <= 0: skill_range = 6000
        
        hx, hz = self._pos(hero); ex, ez = self._pos(enemy)
        dist = math.hypot(hx - ex, hz - ez)
        out.append(1.0 if dist <= skill_range else 0.0)

    def skill_3_enemy_in_range(self, hero, out, _):
        """技能3范围内是否有敌人"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        skill_range = _f(_safe_get(hero, "skill_state", "skill_3_range", default=7000))
        if skill_range <= 0: skill_range = 7000
        
        hx, hz = self._pos(hero); ex, ez = self._pos(enemy)
        dist = math.hypot(hx - ex, hz - ez)
        out.append(1.0 if dist <= skill_range else 0.0)

    def auto_attack_enemy_in_range(self, hero, out, _):
        """普攻范围内是否有敌人"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 使用已有的attack_range
        attack_range = _f(_safe_get(hero, "actor_state", "attack_range", default=4000))
        
        hx, hz = self._pos(hero); ex, ez = self._pos(enemy)
        dist = math.hypot(hx - ex, hz - ez)
        out.append(1.0 if dist <= attack_range else 0.0)

    def displacement_skill_available(self, hero, out, _):
        """位移技能可用性（通常是技能2或3）"""
        if hero is None: out.append(0.0); return
        
        s = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        # 假设技能2是位移技能（可根据实际英雄调整）
        if len(s) > 1:
            displacement_ready = s[1].get("usable", False)
            out.append(1.0 if displacement_ready else 0.0)
        else:
            out.append(0.0)

    def ultimate_combo_ready(self, hero, out, _):
        """大招连击就绪状态"""
        if hero is None: out.append(0.0); return
        
        s = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        # 大招通常是技能3
        if len(s) > 2:
            ult_ready = s[2].get("usable", False)
            out.append(1.0 if ult_ready else 0.0)
        else:
            out.append(0.0)

    def enemy_skill_threat_level(self, hero, out, _):
        """敌方技能威胁等级"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 计算敌方可用技能数量
        s = _safe_get(enemy, "skill_state", "slot_states", default=[]) or []
        available_skills = sum(1 for i in range(min(4, len(s))) if s[i].get("usable", False))
        threat_level = min(available_skills / 4.0, 1.0)  # 归一化到[0,1]
        out.append(threat_level)

    def dodge_window_available(self, hero, out, _):
        """躲避窗口可用性"""
        if hero is None: out.append(0.0); return
        
        # 基于移动速度和位移技能判断躲避能力
        move_speed = self._panel(hero, "mov_spd")
        s = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        has_displacement = len(s) > 1 and s[1].get("usable", False)
        
        # 躲避窗口 = 移动速度权重 + 位移技能权重
        dodge_score = min(move_speed / 1000.0, 0.7) + (0.3 if has_displacement else 0.0)
        out.append(min(dodge_score, 1.0))

    def engagement_advantage(self, hero, out, _):
        """交战优势度评估"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 综合评估：血量比例 + 技能可用性 + 装备优势
        my_hp_ratio = _f(_safe_get(hero, "actor_state", "hp", default=0)) / max(1, _f(_safe_get(hero, "actor_state", "max_hp", default=1)))
        enemy_hp_ratio = _f(_safe_get(enemy, "actor_state", "hp", default=0)) / max(1, _f(_safe_get(enemy, "actor_state", "max_hp", default=1)))
        
        hp_advantage = my_hp_ratio - enemy_hp_ratio
        
        # 技能优势
        my_skills = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        enemy_skills = _safe_get(enemy, "skill_state", "slot_states", default=[]) or []
        my_skill_count = sum(1 for i in range(min(4, len(my_skills))) if my_skills[i].get("usable", False))
        enemy_skill_count = sum(1 for i in range(min(4, len(enemy_skills))) if enemy_skills[i].get("usable", False))
        skill_advantage = (my_skill_count - enemy_skill_count) / 4.0
        
        # 综合优势
        total_advantage = (hp_advantage * 0.6 + skill_advantage * 0.4)
        out.append(max(-1.0, min(1.0, total_advantage)))

    def counter_attack_window(self, hero, out, _):
        """反击窗口可用性"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 基于敌方技能CD状态判断反击窗口
        enemy_skills = _safe_get(enemy, "skill_state", "slot_states", default=[]) or []
        skills_on_cd = sum(1 for i in range(min(3, len(enemy_skills))) if not enemy_skills[i].get("usable", False))
        
        # 敌方技能越多在CD，反击窗口越好
        counter_window = skills_on_cd / 3.0
        out.append(counter_window)

    # 114..123: Combat Distance & Position Features (10 dims)
    def optimal_fight_distance(self, hero, out, _):
        """最优交战距离评估"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        my_attack_range = _f(_safe_get(hero, "actor_state", "attack_range", default=4000))
        enemy_attack_range = _f(_safe_get(enemy, "actor_state", "attack_range", default=4000))
        current_distance = math.hypot(*[a-b for a,b in zip(self._pos(hero), self._pos(enemy))])
        
        # 最优距离：在自己攻击范围内，但尽量超出敌方攻击范围
        if my_attack_range > enemy_attack_range:
            optimal_dist = (my_attack_range + enemy_attack_range) / 2.0
        else:
            optimal_dist = my_attack_range * 0.9  # 稍微保守一点
        
        # 计算当前距离与最优距离的偏差
        distance_score = 1.0 - min(abs(current_distance - optimal_dist) / optimal_dist, 1.0)
        out.append(distance_score)

    def kite_distance_advantage(self, hero, out, _):
        """风筝距离优势"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        my_attack_range = _f(_safe_get(hero, "actor_state", "attack_range", default=4000))
        enemy_attack_range = _f(_safe_get(enemy, "actor_state", "attack_range", default=4000))
        my_speed = self._panel(hero, "mov_spd")
        enemy_speed = self._panel(enemy, "mov_spd")
        
        # 风筝优势 = 射程优势 + 速度优势
        range_advantage = (my_attack_range - enemy_attack_range) / 8000.0  # 归一化
        speed_advantage = (my_speed - enemy_speed) / 1000.0  # 归一化
        
        kite_advantage = (range_advantage * 0.7 + speed_advantage * 0.3)
        out.append(max(-1.0, min(1.0, kite_advantage)))

    def escape_route_available(self, hero, out, _):
        """逃跑路径可用性"""
        if hero is None: out.append(0.0); return
        
        # 基于位移技能和移动速度评估逃跑能力
        s = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        has_displacement = any(s[i].get("usable", False) for i in range(1, min(4, len(s))))
        move_speed = self._panel(hero, "mov_spd")
        
        # 逃跑评分
        escape_score = min(move_speed / 1000.0, 0.6) + (0.4 if has_displacement else 0.0)
        out.append(min(escape_score, 1.0))

    def chase_potential(self, hero, out, _):
        """追击潜力"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        my_speed = self._panel(hero, "mov_spd")
        enemy_speed = self._panel(enemy, "mov_spd")
        
        # 追击能力主要看速度差异和位移技能
        s = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        has_gap_closer = any(s[i].get("usable", False) for i in range(1, min(4, len(s))))
        
        speed_advantage = (my_speed - enemy_speed) / 1000.0
        chase_score = speed_advantage + (0.3 if has_gap_closer else 0.0)
        out.append(max(0.0, min(1.0, chase_score)))

    def terrain_advantage(self, hero, out, _):
        """地形优势"""
        if hero is None: out.append(0.0); return
        
        # 基于草丛状态和位置判断地形优势
        in_grass = _b(_safe_get(hero, "isInGrass", default=False) or _safe_get(hero, "actor_state", "isInGrass", default=False))
        
        # 简单的地形优势评估（可以根据具体地图扩展）
        terrain_score = 0.5 if in_grass else 0.0
        out.append(terrain_score)

    def wall_distance_factor(self, hero, out, _):
        """距离墙体的战术距离"""
        if hero is None: out.append(0.0); return
        
        # 简化实现：基于位置估算到墙的距离
        hx, hz = self._pos(hero)
        
        # 假设地图边界在±30000范围内
        dist_to_wall = min(
            abs(30000 - abs(hx)),  # 到左右墙的距离
            abs(30000 - abs(hz))   # 到上下墙的距离
        )
        
        # 归一化：距离墙越近，战术选择越受限
        wall_factor = min(dist_to_wall / 5000.0, 1.0)
        out.append(wall_factor)

    def retreat_path_safety(self, hero, out, _):
        """撤退路径安全性"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(1.0); return
        
        # 基于与敌人的相对位置和自身状态评估撤退安全性
        hx, hz = self._pos(hero)
        ex, ez = self._pos(enemy)
        
        # 计算朝向己方基地的撤退方向的安全性
        # 假设己方基地在负坐标方向（可根据阵营调整）
        if self.mirror:
            base_direction_x, base_direction_z = 1.0, 0.0
        else:
            base_direction_x, base_direction_z = -1.0, 0.0
        
        # 撤退方向向量
        retreat_dx = base_direction_x
        retreat_dz = base_direction_z
        
        # 敌人相对位置
        enemy_dx = ex - hx
        enemy_dz = ez - hz
        enemy_dist = math.hypot(enemy_dx, enemy_dz)
        
        if enemy_dist > 0:
            enemy_dx /= enemy_dist
            enemy_dz /= enemy_dist
            
            # 计算敌人是否在撤退路径上（点积）
            enemy_on_retreat_path = retreat_dx * enemy_dx + retreat_dz * enemy_dz
            
            # 撤退安全性：敌人越不在撤退路径上越安全
            safety = 1.0 - max(0.0, enemy_on_retreat_path)
        else:
            safety = 0.0
        
        out.append(safety)

    def flanking_opportunity(self, hero, out, _):
        """侧翼机会"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 基于敌人的朝向和自己的位置判断侧翼机会
        ex, ez = self._pos(enemy)
        hx, hz = self._pos(hero)
        
        # 获取敌人朝向
        enemy_fx, enemy_fz = self._forward(enemy)
        
        # 计算自己相对于敌人的位置向量
        to_me_x = hx - ex
        to_me_z = hz - ez
        to_me_dist = math.hypot(to_me_x, to_me_z)
        
        if to_me_dist > 0:
            to_me_x /= to_me_dist
            to_me_z /= to_me_dist
            
            # 计算自己是否在敌人的侧面或背后
            dot_product = enemy_fx * to_me_x + enemy_fz * to_me_z
            
            # dot_product < 0 表示在敌人背后，接近0表示在侧面
            flank_score = max(0.0, -dot_product)  # 越在背后分数越高
        else:
            flank_score = 0.0
        
        out.append(flank_score)

    def resource_advantage_ratio(self, hero, out, _):
        """综合资源优势比例"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # HP优势
        my_hp = _f(_safe_get(hero, "actor_state", "hp", default=0))
        my_max_hp = _f(_safe_get(hero, "actor_state", "max_hp", default=1))
        enemy_hp = _f(_safe_get(enemy, "actor_state", "hp", default=0))
        enemy_max_hp = _f(_safe_get(enemy, "actor_state", "max_hp", default=1))
        
        my_hp_ratio = my_hp / max(1, my_max_hp)
        enemy_hp_ratio = enemy_hp / max(1, enemy_max_hp)
        hp_advantage = my_hp_ratio - enemy_hp_ratio
        
        # MP优势
        my_ep = _f(_safe_get(hero, "actor_state", "values", default={}).get("ep", 0))
        my_max_ep = _f(_safe_get(hero, "actor_state", "values", default={}).get("max_ep", 1))
        enemy_ep = _f(_safe_get(enemy, "actor_state", "values", default={}).get("ep", 0))
        enemy_max_ep = _f(_safe_get(enemy, "actor_state", "values", default={}).get("max_ep", 1))
        
        my_ep_ratio = my_ep / max(1, my_max_ep)
        enemy_ep_ratio = enemy_ep / max(1, enemy_max_ep)
        ep_advantage = my_ep_ratio - enemy_ep_ratio
        
        # 综合资源优势
        resource_advantage = hp_advantage * 0.7 + ep_advantage * 0.3
        out.append(max(-1.0, min(1.0, resource_advantage)))

    def all_in_threshold(self, hero, out, _):
        """全力进攻阈值评估"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 基于敌方血量和己方优势判断是否应该全力进攻
        enemy_hp = _f(_safe_get(enemy, "actor_state", "hp", default=0))
        enemy_max_hp = _f(_safe_get(enemy, "actor_state", "max_hp", default=1))
        enemy_hp_ratio = enemy_hp / max(1, enemy_max_hp)
        
        # 己方技能就绪情况
        s = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        skills_ready = sum(1 for i in range(min(4, len(s))) if s[i].get("usable", False))
        skill_readiness = skills_ready / 4.0
        
        # 全力进攻条件：敌方血量低 + 己方技能就绪
        all_in_score = (1.0 - enemy_hp_ratio) * 0.6 + skill_readiness * 0.4
        out.append(all_in_score)

    # 124..131: Advanced Tactical Prediction Features (8 dims)
    def enemy_next_action_predict(self, hero, out, _):
        """敌方下一步行动预测"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 基于敌方状态预测其行为倾向
        enemy_hp = _f(_safe_get(enemy, "actor_state", "hp", default=0))
        enemy_max_hp = _f(_safe_get(enemy, "actor_state", "max_hp", default=1))
        enemy_hp_ratio = enemy_hp / max(1, enemy_max_hp)
        
        # 敌方技能状态
        enemy_skills = _safe_get(enemy, "skill_state", "slot_states", default=[]) or []
        skills_ready = sum(1 for i in range(min(3, len(enemy_skills))) if enemy_skills[i].get("usable", False))
        
        # 距离因素
        hx, hz = self._pos(hero); ex, ez = self._pos(enemy)
        distance = math.hypot(hx - ex, hz - ez)
        
        # 预测逻辑：
        # 高血量+技能就绪+近距离 -> 攻击倾向 (0.8)
        # 低血量+远距离 -> 撤退倾向 (0.2)
        # 其他 -> 观望/走位 (0.5)
        if enemy_hp_ratio > 0.6 and skills_ready >= 2 and distance < 6000:
            action_predict = 0.8  # 攻击倾向
        elif enemy_hp_ratio < 0.4 and distance > 8000:
            action_predict = 0.2  # 撤退倾向
        else:
            action_predict = 0.5  # 中性/走位
        
        out.append(action_predict)

    def burst_combo_window(self, hero, out, _):
        """爆发连击窗口"""
        if hero is None: out.append(0.0); return
        
        # 检查己方爆发连击的条件
        s = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        
        # 核心技能就绪（假设技能1、3是关键）
        skill1_ready = len(s) > 0 and s[0].get("usable", False)
        skill3_ready = len(s) > 2 and s[2].get("usable", False)
        
        # 血量和蓝量充足
        hp_ratio = _f(_safe_get(hero, "actor_state", "hp", default=0)) / max(1, _f(_safe_get(hero, "actor_state", "max_hp", default=1)))
        ep_ratio = _f(_safe_get(hero, "actor_state", "values", default={}).get("ep", 0)) / max(1, _f(_safe_get(hero, "actor_state", "values", default={}).get("max_ep", 1)))
        
        # 爆发窗口评分
        combo_score = 0.0
        if skill1_ready and skill3_ready:  # 关键技能就绪
            combo_score += 0.5
        if hp_ratio > 0.5:  # 血量安全
            combo_score += 0.3
        if ep_ratio > 0.6:  # 蓝量充足
            combo_score += 0.2
        
        out.append(combo_score)

    def defensive_stance_value(self, hero, out, _):
        """防守姿态价值"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 基于相对实力判断防守价值
        my_hp_ratio = _f(_safe_get(hero, "actor_state", "hp", default=0)) / max(1, _f(_safe_get(hero, "actor_state", "max_hp", default=1)))
        enemy_hp_ratio = _f(_safe_get(enemy, "actor_state", "hp", default=0)) / max(1, _f(_safe_get(enemy, "actor_state", "max_hp", default=1)))
        
        # 技能差距
        my_skills = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        enemy_skills = _safe_get(enemy, "skill_state", "slot_states", default=[]) or []
        my_skill_count = sum(1 for i in range(min(4, len(my_skills))) if my_skills[i].get("usable", False))
        enemy_skill_count = sum(1 for i in range(min(4, len(enemy_skills))) if enemy_skills[i].get("usable", False))
        
        # 防守价值：己方劣势时防守价值高
        hp_disadvantage = enemy_hp_ratio - my_hp_ratio
        skill_disadvantage = (enemy_skill_count - my_skill_count) / 4.0
        
        defensive_value = (hp_disadvantage * 0.6 + skill_disadvantage * 0.4)
        out.append(max(0.0, min(1.0, defensive_value)))

    def zone_control_advantage(self, hero, out, _):
        """区域控制优势"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 基于射程和位置判断区域控制能力
        my_attack_range = _f(_safe_get(hero, "actor_state", "attack_range", default=4000))
        enemy_attack_range = _f(_safe_get(enemy, "actor_state", "attack_range", default=4000))
        
        # 射程优势
        range_advantage = (my_attack_range - enemy_attack_range) / 8000.0
        
        # 位置优势（基于地形，这里简化）
        hx, hz = self._pos(hero)
        in_grass = _b(_safe_get(hero, "isInGrass", default=False) or _safe_get(hero, "actor_state", "isInGrass", default=False))
        position_advantage = 0.3 if in_grass else 0.0
        
        zone_control = range_advantage * 0.7 + position_advantage
        out.append(max(-1.0, min(1.0, zone_control)))

    def timing_critical_moment(self, hero, out, _):
        """关键时机判断"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 判断当前是否是关键时机
        critical_factors = 0.0
        
        # 1. 敌方技能空窗期
        enemy_skills = _safe_get(enemy, "skill_state", "slot_states", default=[]) or []
        skills_on_cd = sum(1 for i in range(min(3, len(enemy_skills))) if not enemy_skills[i].get("usable", False))
        if skills_on_cd >= 2:  # 敌方多个技能CD
            critical_factors += 0.4
        
        # 2. 己方技能就绪
        my_skills = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        my_skills_ready = sum(1 for i in range(min(3, len(my_skills))) if my_skills[i].get("usable", False))
        if my_skills_ready >= 2:  # 己方技能就绪
            critical_factors += 0.3
        
        # 3. 血量差距适合决胜
        my_hp_ratio = _f(_safe_get(hero, "actor_state", "hp", default=0)) / max(1, _f(_safe_get(hero, "actor_state", "max_hp", default=1)))
        enemy_hp_ratio = _f(_safe_get(enemy, "actor_state", "hp", default=0)) / max(1, _f(_safe_get(enemy, "actor_state", "max_hp", default=1)))
        
        if 0.3 < enemy_hp_ratio < 0.7 and my_hp_ratio > 0.5:  # 敌方血量可击杀范围
            critical_factors += 0.3
        
        out.append(min(critical_factors, 1.0))

    def micro_positioning_score(self, hero, out, _):
        """微操走位评分"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        hx, hz = self._pos(hero); ex, ez = self._pos(enemy)
        distance = math.hypot(hx - ex, hz - ez)
        
        my_attack_range = _f(_safe_get(hero, "actor_state", "attack_range", default=4000))
        enemy_attack_range = _f(_safe_get(enemy, "actor_state", "attack_range", default=4000))
        
        # 理想的微操走位：
        # 1. 在自己攻击范围边缘
        # 2. 超出敌方攻击范围
        # 3. 有足够的走位空间
        
        # 距离评分
        if my_attack_range > enemy_attack_range:
            # 远程优势，保持风筝距离
            ideal_distance = (my_attack_range + enemy_attack_range) / 2.0
            distance_score = 1.0 - min(abs(distance - ideal_distance) / ideal_distance, 1.0)
        else:
            # 近战，需要贴身
            ideal_distance = my_attack_range * 0.8
            distance_score = 1.0 - min(abs(distance - ideal_distance) / ideal_distance, 1.0)
        
        # 走位空间（基于到墙体的距离）
        wall_space = min(
            abs(30000 - abs(hx)),
            abs(30000 - abs(hz))
        )
        space_score = min(wall_space / 5000.0, 1.0)
        
        positioning_score = distance_score * 0.7 + space_score * 0.3
        out.append(positioning_score)

    def risk_reward_ratio(self, hero, out, _):
        """风险收益比"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 评估当前行动的风险收益比
        my_hp = _f(_safe_get(hero, "actor_state", "hp", default=0))
        my_max_hp = _f(_safe_get(hero, "actor_state", "max_hp", default=1))
        enemy_hp = _f(_safe_get(enemy, "actor_state", "hp", default=0))
        enemy_max_hp = _f(_safe_get(enemy, "actor_state", "max_hp", default=1))
        
        my_hp_ratio = my_hp / max(1, my_max_hp)
        enemy_hp_ratio = enemy_hp / max(1, enemy_max_hp)
        
        # 收益：击杀敌方的可能性
        kill_reward = 1.0 - enemy_hp_ratio
        
        # 风险：被敌方击杀的风险
        death_risk = 1.0 - my_hp_ratio
        
        # 技能状态影响风险收益
        my_skills = _safe_get(hero, "skill_state", "slot_states", default=[]) or []
        enemy_skills = _safe_get(enemy, "skill_state", "slot_states", default=[]) or []
        my_skill_count = sum(1 for i in range(min(4, len(my_skills))) if my_skills[i].get("usable", False))
        enemy_skill_count = sum(1 for i in range(min(4, len(enemy_skills))) if enemy_skills[i].get("usable", False))
        
        skill_advantage = (my_skill_count - enemy_skill_count) / 4.0
        
        # 综合风险收益比
        if death_risk > 0.1:
            risk_reward = (kill_reward + skill_advantage * 0.5) / death_risk
        else:
            risk_reward = kill_reward + skill_advantage * 0.5
        
        out.append(max(0.0, min(2.0, risk_reward)) / 2.0)  # 归一化到[0,1]

    def prediction_confidence(self, hero, out, _):
        """预测置信度"""
        if hero is None: out.append(0.0); return
        enemy = self._enemy(hero)
        if not enemy: out.append(0.0); return
        
        # 基于信息完整性评估预测的置信度
        confidence_factors = 0.0
        
        # 1. 视野完整性（敌方是否在视野中）
        hx, hz = self._pos(hero); ex, ez = self._pos(enemy)
        distance = math.hypot(hx - ex, hz - ez)
        if distance < 8000:  # 在视野范围内
            confidence_factors += 0.4
        
        # 2. 状态信息完整性
        enemy_skills = _safe_get(enemy, "skill_state", "slot_states", default=[]) or []
        if len(enemy_skills) >= 3:  # 技能信息完整
            confidence_factors += 0.3
        
        # 3. 历史行为一致性（简化）
        confidence_factors += 0.3  # 假设有一定的历史数据支持
        
        out.append(min(confidence_factors, 1.0))