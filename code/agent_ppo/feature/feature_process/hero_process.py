#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors (extended)

Hero feature processing (1v1) — optimized baseline focusing on
FOUNDATIONAL but IMPORTANT signals aligned with official observations:
- Survival & geometry (alive, pos/dir, hp/ep rates + raw hp/ep/max)
- Economy & respawn: level, money, exp, revive_time, K/D
- Panels: atk/def/move/atkspd/crit/penetration/vamp/cd_reduce/ctrl_reduce/recover
- Skill slots: usable & cooldown ratio, usage/hit (cum + recent deltas), simple combo window
- Combat geometry: distance & facing, mutual attack-range checks, lock/visibility
- Common-attack readiness & enhanced-common awareness (buff based)
- Kiting cues: closing_speed / strafe_cos / time_since_last_common
- DPS & simple TTK proxies
- TOWER RISK TRIO (NEW): in_my_tower_range, in_enemy_tower_range, under_enemy_tower_fire
  (derived from npc_states (towers) + bullets when available)
NOTE: Per user request, we DO NOT include summoner-skill or active-equip features.

Interfaces preserved:
- get_hero_config() reads hero_feature_config.ini
- process_vec_hero(frame_state) -> concatenated [my_hero_vec + enemy_hero_vec]
- generate_hero_info_dict(), generate_hero_info_list(), generate_one_type_hero_feature()
- Feature names & normalizations are driven by INI [feature_config]/[feature_functions]
"""
from enum import Enum
from collections import OrderedDict, defaultdict, deque
import configparser
import math
import os

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer


# ------------------------------ Utilities ------------------------------ #
def _float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


class HeroProcess:
    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.transform_camp2_to_camp1 = (camp == "PLAYERCAMP_2")

        # parsed from INI
        self.feature_order = []       # ordered feature names from [feature_functions]
        self.feature_func_map = {}    # feature_name -> bound getter
        self.map_feature_to_norm = {} # feature_name -> (norm_func, *params)

        # caches
        self._frame_bullets = []
        self._frame_npcs = []  # towers included
        self._main_hero_dict = OrderedDict()
        self._enemy_hero_dict = OrderedDict()

        # cross-frame memory for deltas/windows
        self._mem_last_pos = {}     # hero_id -> (x,z)
        self._mem_last_dist = {}    # (my_id, opp_id) -> last distance
        self._mem_last_common = {}  # hero_id -> frames_since_last_common
        self._mem_slot_used = defaultdict(lambda: defaultdict(int))
        self._mem_slot_hit  = defaultdict(lambda: defaultdict(int))
        self._mem_recent_win = defaultdict(lambda: defaultdict(lambda: {
            "used": deque(maxlen=30), "hit": deque(maxlen=30)
        }))
        self._mem_post_s2_pulse = defaultdict(int)  # 2->1->AA small window
        self._mem_enemy_bullets_in = defaultdict(lambda: deque(maxlen=10))

        # constants
        self.view_dist = 16000.0  # for bullet proximity

        self.get_hero_config()  # read INI

    def reset(self, camp):
        self.__init__(camp)

    # ------------------------------------------------------------------ #
    # Config parsing
    # ------------------------------------------------------------------ #
    def get_hero_config(self):
        cfg = configparser.ConfigParser()
        cfg.optionxform = str
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir, "hero_feature_config.ini")
        if not os.path.exists(path):
            raise FileNotFoundError(f"hero_feature_config.ini not found at {path}")

        cfg.read(path)

        # normalization table
        hero_feature_config = []
        for fname, spec in cfg["feature_config"].items():
            hero_feature_config.append(f"{fname}:{spec}")
        self.map_feature_to_norm = self.normalizer.parse_config(hero_feature_config)

        # ordered feature functions
        self.feature_order.clear()
        self.feature_func_map.clear()
        for fname, getter in cfg["feature_functions"].items():
            self.feature_order.append(fname)
            if not hasattr(self, getter):
                raise ValueError(f"Unsupported getter function: {getter} for feature {fname}")
            self.feature_func_map[fname] = getattr(self, getter)

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #
    def process_vec_hero(self, frame_state: dict):
        # per-frame caches
        self._frame_bullets = list(frame_state.get("bullets", []) or [])
        self._frame_npcs = list(frame_state.get("npc_states", []) or [])
        self.generate_hero_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)  # kept for compatibility

        main_vec = self.generate_one_type_hero_feature(self._main_hero_dict, "main_camp")
        enemy_vec = self.generate_one_type_hero_feature(self._enemy_hero_dict, "enemy_camp")
        return main_vec + enemy_vec

    # ------------------------------------------------------------------ #
    # Hero dict builders (1v1: one hero per camp)
    # ------------------------------------------------------------------ #
    def generate_hero_info_dict(self, frame_state):
        self._main_hero_dict.clear()
        self._enemy_hero_dict.clear()
        heros = frame_state.get("hero_states", []) or []
        for h in heros:
            ast = h.get("actor_state", {}) or {}
            if _float(ast.get("hp", 0.0)) <= 0.0 and ast.get("behav_mode", "") == "ObjBehaviMode_Dead":
                # keep dead heroes as well for revive_time etc.
                pass
            camp = ast.get("camp")
            rid = _int(ast.get("runtime_id", ast.get("config_id", -1)), -1)
            if camp == self.main_camp:
                self._main_hero_dict[rid] = h
            else:
                self._enemy_hero_dict[rid] = h

        # stable order
        self._main_hero_dict = OrderedDict(sorted(self._main_hero_dict.items()))
        self._enemy_hero_dict = OrderedDict(sorted(self._enemy_hero_dict.items()))

    def generate_hero_info_list(self, frame_state):
        # For compatibility — not used in this implementation
        return

    # ------------------------------------------------------------------ #
    # Vectorize one side according to INI order
    # ------------------------------------------------------------------ #
    def generate_one_type_hero_feature(self, hero_dict, camp_tag: str):
        vec = []
        hero = next(iter(hero_dict.values()), None)
        if hero is None:
            # pad zeros by applying normalizer on 0 for each feature
            for fname in self.feature_order:
                norm_entry = self.map_feature_to_norm.get(fname)
                if not norm_entry:
                    vec.append(0.0); continue
                norm_func, *params = norm_entry
                v = norm_func(0.0, *params)
                if isinstance(v, list): vec.extend(v)
                else: vec.append(v)
            return vec

        # expand in configured order
        for fname in self.feature_order:
            getter = self.feature_func_map[fname]
            raw_vals = []
            getter(hero, raw_vals, fname)  # fill raw_vals (list)
            norm_entry = self.map_feature_to_norm.get(fname)
            if not norm_entry:
                raise KeyError(f"Feature '{fname}' missing in normalizer map")
            norm_func, *params = norm_entry
            for rv in raw_vals:
                nv = norm_func(rv, *params)
                if isinstance(nv, list):
                    vec.extend(nv)
                else:
                    vec.append(nv)
        return vec

    # ================================================================== #
    # =============== Basic getters (survival / geometry) =============== #
    # ================================================================== #
    def is_alive(self, hero, out, feature_name):
        hp = _float((hero.get("actor_state") or {}).get("hp", 0.0))
        out.append(1.0 if hp > 0.0 else 0.0)

    def get_location_x(self, hero, out, feature_name):
        x = _float((hero.get("actor_state") or {}).get("location", {}).get("x", 0.0))
        if self.transform_camp2_to_camp1: x = -x
        out.append(x)

    def get_location_z(self, hero, out, feature_name):
        z = _float((hero.get("actor_state") or {}).get("location", {}).get("z", 0.0))
        if self.transform_camp2_to_camp1: z = -z
        out.append(z)

    def get_forward_x(self, hero, out, feature_name):
        fx = _float((hero.get("actor_state") or {}).get("forward", {}).get("x", 0.0))
        fz = _float((hero.get("actor_state") or {}).get("forward", {}).get("z", 0.0))
        n = math.hypot(fx, fz) + 1e-8
        v = (fx / n) if n > 0 else 0.0
        if self.transform_camp2_to_camp1: v = -v
        out.append(v)

    def get_forward_z(self, hero, out, feature_name):
        fx = _float((hero.get("actor_state") or {}).get("forward", {}).get("x", 0.0))
        fz = _float((hero.get("actor_state") or {}).get("forward", {}).get("z", 0.0))
        n = math.hypot(fx, fz) + 1e-8
        v = (fz / n) if n > 0 else 0.0
        if self.transform_camp2_to_camp1: v = -v
        out.append(v)

    def get_hp_rate(self, hero, out, feature_name):
        ast = hero.get("actor_state", {}) or {}
        out.append(0.0 if _float(ast.get("max_hp", 1.0)) <= 0 else _float(ast.get("hp", 0.0)) / _float(ast.get("max_hp", 1.0)))

    def get_ep_rate(self, hero, out, feature_name):
        vals = (hero.get("actor_state") or {}).get("values", {}) or {}
        out.append(0.0 if _float(vals.get("max_ep", 1.0)) <= 0 else _float(vals.get("ep", 0.0)) / _float(vals.get("max_ep", 1.0)))

    def get_level(self, hero, out, feature_name): out.append(_float(hero.get("level", 0)))
    def get_exp(self, hero, out, feature_name): out.append(_float(hero.get("exp", 0)))
    def get_money(self, hero, out, feature_name): out.append(_float(hero.get("money", 0)))

    def get_hp_raw(self, hero, out, feature_name): out.append(_float((hero.get("actor_state") or {}).get("hp", 0.0)))
    def get_max_hp(self, hero, out, feature_name): out.append(_float((hero.get("actor_state") or {}).get("max_hp", 1.0)))
    def get_ep_raw(self, hero, out, feature_name): out.append(_float(((hero.get("actor_state") or {}).get("values") or {}).get("ep", 0.0)))
    def get_max_ep(self, hero, out, feature_name): out.append(_float(((hero.get("actor_state") or {}).get("values") or {}).get("max_ep", 1.0)))

    def get_attack_range(self, hero, out, feature_name):
        out.append(_float((hero.get("actor_state") or {}).get("attack_range", 0.0)))

    def get_is_in_grass(self, hero, out, feature_name):
        out.append(1.0 if bool(hero.get("isInGrass", False)) else 0.0)

    def get_revive_time(self, hero, out, feature_name):
        out.append(_float(hero.get("revive_time", 0)))

    def get_kill_cnt(self, hero, out, feature_name): out.append(_float(hero.get("killCnt", 0)))
    def get_dead_cnt(self, hero, out, feature_name): out.append(_float(hero.get("deadCnt", 0)))

    # ------------------ Panel values ------------------ #
    def _val(self, hero, key, default=0.0):
        return _float(((hero.get("actor_state") or {}).get("values") or {}).get(key, default))

    def get_phy_atk(self, hero, out, feature_name): out.append(self._val(hero, "phy_atk"))
    def get_phy_def(self, hero, out, feature_name): out.append(self._val(hero, "phy_def"))
    def get_mgc_atk(self, hero, out, feature_name): out.append(self._val(hero, "mgc_atk"))
    def get_mgc_def(self, hero, out, feature_name): out.append(self._val(hero, "mgc_def"))
    def get_mov_spd(self, hero, out, feature_name): out.append(self._val(hero, "mov_spd"))
    def get_atk_spd(self, hero, out, feature_name): out.append(self._val(hero, "atk_spd"))
    def get_crit_rate(self, hero, out, feature_name): out.append(self._val(hero, "crit_rate"))
    def get_crit_effe(self, hero, out, feature_name): out.append(self._val(hero, "crit_effe"))
    def get_phy_armor_hurt(self, hero, out, feature_name): out.append(self._val(hero, "phy_armor_hurt"))
    def get_mgc_armor_hurt(self, hero, out, feature_name): out.append(self._val(hero, "mgc_armor_hurt"))
    def get_phy_vamp(self, hero, out, feature_name): out.append(self._val(hero, "phy_vamp"))
    def get_mgc_vamp(self, hero, out, feature_name): out.append(self._val(hero, "mgc_vamp"))
    def get_cd_reduce(self, hero, out, feature_name): out.append(self._val(hero, "cd_reduce"))
    def get_ctrl_reduce(self, hero, out, feature_name): out.append(self._val(hero, "ctrl_reduce"))
    def get_hp_recover(self, hero, out, feature_name): out.append(self._val(hero, "hp_recover"))
    def get_ep_recover(self, hero, out, feature_name): out.append(self._val(hero, "ep_recover"))

    # ================================================================== #
    # ==================== Skill slot based features =================== #
    # ================================================================== #
    def _slot(self, hero, idx: int):
        slots = (hero.get("skill_state") or {}).get("slot_states", []) or []
        if 0 <= idx < len(slots):
            return slots[idx]
        return None

    def get_skill_cd_rate(self, hero, out, feature_name):
        idx = self._digits(feature_name)
        slot = self._slot(hero, idx)
        if not slot: out.append(0.0); return
        cd = _float(slot.get("cooldown", 0.0)); mx = _float(slot.get("cooldown_max", 0.0))
        out.append(0.0 if mx <= 1e-6 else cd / mx)

    def get_skill_usable(self, hero, out, feature_name):
        idx = self._digits(feature_name)
        slot = self._slot(hero, idx)
        out.append(1.0 if (slot and bool(slot.get("usable", False))) else 0.0)

    def get_skill_usedTimes(self, hero, out, feature_name):
        idx = self._digits(feature_name)
        slot = self._slot(hero, idx) or {}
        out.append(_float(slot.get("usedTimes", 0)))

    def get_skill_hit_rate(self, hero, out, feature_name):
        idx = self._digits(feature_name)
        slot = self._slot(hero, idx) or {}
        used = _int(slot.get("usedTimes", 0))
        hit  = _int(slot.get("hitHeroTimes", 0))
        out.append(0.0 if used <= 0 else (hit / max(1, used)))

    def get_slot_used_delta(self, hero, out, feature_name):
        idx = self._digits(feature_name)
        slot = self._slot(hero, idx) or {}
        used = _int(slot.get("usedTimes", 0))
        hid = self._rid(hero)
        last = self._mem_slot_used[hid][idx]
        delta = 1.0 if used > last else 0.0
        self._mem_slot_used[hid][idx] = used
        self._mem_recent_win[hid][idx]["used"].append(1 if delta > 0.5 else 0)
        # pulse a tiny window after using skill2
        if idx == 2 and delta > 0.5:
            self._mem_post_s2_pulse[hid] = 2
        out.append(delta)

    def get_slot_hit_delta(self, hero, out, feature_name):
        idx = self._digits(feature_name)
        slot = self._slot(hero, idx) or {}
        hit = _int(slot.get("hitHeroTimes", 0))
        hid = self._rid(hero)
        last = self._mem_slot_hit[hid][idx]
        delta = 1.0 if hit > last else 0.0
        self._mem_slot_hit[hid][idx] = hit
        self._mem_recent_win[hid][idx]["hit"].append(1 if delta > 0.5 else 0)
        out.append(delta)

    def get_slot_recent_hit_rate(self, hero, out, feature_name):
        idx = self._digits(feature_name); hid = self._rid(hero)
        q_used = self._mem_recent_win[hid][idx]["used"]
        q_hit  = self._mem_recent_win[hid][idx]["hit"]
        s_used = sum(q_used); s_hit = sum(q_hit)
        out.append(0.0 if s_used <= 0 else (s_hit / max(1, s_used)))

    def get_combo_effect_time_max(self, hero, out, feature_name):
        slots = (hero.get("skill_state") or {}).get("slot_states", []) or []
        cmax = 0.0
        for s in slots:
            cmax = max(cmax, _float(s.get("comboEffectTime", 0.0)))
        out.append(cmax)

    # Common-attack readiness (explicit)
    def get_common_skill_is_useable(self, hero, out, feature_name):
        ast = hero.get("actor_state", {}) or {}
        val = ast.get("common_skill_is_useable")
        if val is None:
            out.append(0.0)
        else:
            out.append(1.0 if bool(val) else 0.0)

    # ================================================================== #
    # ==================== Combat geometry / relations ================= #
    # ================================================================== #
    def _rid(self, hero) -> int:
        ast = hero.get("actor_state", {}) or {}
        return _int(ast.get("runtime_id", ast.get("config_id", -1)), -1)

    def _pos(self, hero):
        ast = hero.get("actor_state", {}) or {}
        p = ast.get("location", {}) or {}
        x = _float(p.get("x", 0.0)); z = _float(p.get("z", 0.0))
        if self.transform_camp2_to_camp1: x, z = -x, -z
        return x, z

    def _forward(self, hero):
        ast = hero.get("actor_state", {}) or {}
        f = ast.get("forward", {}) or {}
        fx = _float(f.get("x", 0.0)); fz = _float(f.get("z", 0.0))
        n = math.hypot(fx, fz) + 1e-8
        fx = fx / n if n > 0 else 0.0
        fz = fz / n if n > 0 else 0.0
        if self.transform_camp2_to_camp1: fx, fz = -fx, -fz
        return fx, fz

    def _enemy_of(self, hero):
        my_camp = (hero.get("actor_state") or {}).get("camp")
        # hero belongs to main camp? enemy is the first of enemy dict, else main
        if my_camp == self.main_camp:
            return next(iter(self._enemy_hero_dict.values()), None)
        else:
            return next(iter(self._main_hero_dict.values()), None)

    def get_dist_to_enemy(self, hero, out, feature_name):
        opp = self._enemy_of(hero)
        if not opp: out.append(0.0); return
        x1, z1 = self._pos(hero); x2, z2 = self._pos(opp)
        out.append(math.hypot(x1 - x2, z1 - z2))

    def get_in_my_atk_range(self, hero, out, feature_name):
        opp = self._enemy_of(hero)
        if not opp: out.append(0.0); return
        dist = []; self.get_dist_to_enemy(hero, dist, None)
        my_rng = _float((hero.get("actor_state") or {}).get("attack_range", 0.0))
        out.append(1.0 if (dist and dist[0] <= my_rng) else 0.0)

    def get_in_enemy_atk_range(self, hero, out, feature_name):
        opp = self._enemy_of(hero)
        if not opp: out.append(0.0); return
        x1, z1 = self._pos(hero); x2, z2 = self._pos(opp)
        dist = math.hypot(x1 - x2, z1 - z2)
        his_rng = _float((opp.get("actor_state") or {}).get("attack_range", 0.0))
        out.append(1.0 if dist <= his_rng else 0.0)

    def get_hp_rate_diff(self, hero, out, feature_name):
        opp = self._enemy_of(hero)
        if not opp: out.append(0.0); return
        def hp_rate(h):
            ast = h.get("actor_state", {}) or {}
            return 0.0 if _float(ast.get("max_hp", 1.0)) <= 0 else _float(ast.get("hp", 0.0)) / _float(ast.get("max_hp", 1.0))
        out.append(hp_rate(hero) - hp_rate(opp))

    def get_rel_dir_cos(self, hero, out, feature_name):
        opp = self._enemy_of(hero)
        if not opp: out.append(0.0); return
        x1, z1 = self._pos(hero); x2, z2 = self._pos(opp)
        dx, dz = (x2 - x1), (z2 - z1)
        n = math.hypot(dx, dz) + 1e-8
        ux, uz = dx / n, dz / n
        fx, fz = self._forward(hero)
        out.append(max(-1.0, min(1.0, fx * ux + fz * uz)))

    def get_visible_to_main_camp(self, hero, out, feature_name):
        ast = hero.get("actor_state", {}) or {}
        vis = list(ast.get("camp_visible", []) or [True, True])
        idx = 1 if self.main_camp.endswith("_2") else 0
        out.append(1.0 if (idx < len(vis) and bool(vis[idx])) else 0.0)

    def get_attack_target_is_enemy(self, hero, out, feature_name):
        opp = self._enemy_of(hero)
        if not opp: out.append(0.0); return
        my_ast = hero.get("actor_state", {}) or {}
        tgt = _int(my_ast.get("attack_target", -1), -1)
        opp_id = self._rid(opp)
        out.append(1.0 if (opp_id >= 0 and tgt == opp_id) else 0.0)

    # ================================================================== #
    # ============ Enhanced common-attack awareness & kiting =========== #
    # ================================================================== #
    ENHANCED_COMMON_BUFF_IDS = {111110, 111151}  # example ids; replace with real ones if available

    def get_has_enhanced_common(self, hero, out, feature_name):
        bs = (hero.get("buff_state") or {})
        marks = list(bs.get("buff_marks", []) or [])
        skills = list(bs.get("buff_skills", []) or [])
        ids = set()
        for m in marks:
            cid = m.get("config_id", m.get("configId", m.get("buff_id")))
            if cid is not None: ids.add(_int(cid))
        for s in skills:
            cid = s.get("config_id", s.get("configId", s.get("buff_id")))
            if cid is not None: ids.add(_int(cid))
        out.append(1.0 if any(cid in self.ENHANCED_COMMON_BUFF_IDS for cid in ids) else 0.0)

    def get_enhanced_stack(self, hero, out, feature_name):
        bs = (hero.get("buff_state") or {})
        marks = list(bs.get("buff_marks", []) or [])
        layer = 0
        for m in marks:
            cid = m.get("config_id", m.get("configId", m.get("buff_id")))
            if cid is None: continue
            if _int(cid) in self.ENHANCED_COMMON_BUFF_IDS:
                layer = max(layer, _int(m.get("layer", 1), 1))
        out.append(float(layer))

    def get_enhanced_common_available(self, hero, out, feature_name):
        have = []; self.get_has_enhanced_common(hero, have, None)
        if not have or have[0] < 0.5:
            out.append(0.0); return
        lock, inrng = [], []
        self.get_attack_target_is_enemy(hero, lock, None)
        self.get_in_my_atk_range(hero, inrng, None)
        out.append(1.0 if (lock and inrng and lock[0] >= 0.5 and inrng[0] >= 0.5) else 0.0)

    # Kiting cues
    def get_closing_speed(self, hero, out, feature_name):
        opp = self._enemy_of(hero)
        if not opp: out.append(0.0); return
        my_id = self._rid(hero); opp_id = self._rid(opp)
        x1, z1 = self._pos(hero); x2, z2 = self._pos(opp)
        d = math.hypot(x1 - x2, z1 - z2)
        key = (my_id, opp_id)
        last = self._mem_last_dist.get(key, d)
        self._mem_last_dist[key] = d
        out.append(last - d)  # >0 approaching

    def get_strafe_cos(self, hero, out, feature_name):
        my_id = self._rid(hero)
        x, z = self._pos(hero)
        last = self._mem_last_pos.get(my_id, (x, z))
        vx, vz = x - last[0], z - last[1]
        self._mem_last_pos[my_id] = (x, z)

        opp = self._enemy_of(hero)
        if not opp: out.append(0.0); return
        xo, zo = self._pos(opp)
        dx, dz = (xo - x), (zo - z)
        n = math.hypot(dx, dz) + 1e-8
        ux, uz = dx / n, dz / n
        vn = math.hypot(vx, vz) + 1e-8
        cosv = (vx * ux + vz * uz) / vn
        out.append(max(-1.0, min(1.0, cosv)))

    def get_time_since_last_common(self, hero, out, feature_name):
        my_id = self._rid(hero)
        cnt = self._mem_last_common.get(my_id, 0)
        lock, inrng = [], []
        self.get_attack_target_is_enemy(hero, lock, None)
        self.get_in_my_atk_range(hero, inrng, None)
        if lock and inrng and lock[0] >= 0.5 and inrng[0] >= 0.5:
            cnt = 0
        else:
            cnt = min(cnt + 1, 1_000_000)
        self._mem_last_common[my_id] = cnt
        out.append(float(cnt))

    # DPS / TTK proxies
    def get_expected_crit_factor(self, hero, out, feature_name):
        cr = self._val(hero, "crit_rate") / 10000.0
        ce = self._val(hero, "crit_effe") / 10000.0
        cr = max(0.0, min(1.0, cr)); ce = max(0.0, ce)
        out.append(1.0 + cr * ce)

    def get_dps_proxy(self, hero, out, feature_name):
        phy = self._val(hero, "phy_atk")
        asp = self._val(hero, "atk_spd")
        ecf = []; self.get_expected_crit_factor(hero, ecf, None)
        out.append(phy * ecf[0] * (asp / 1000.0))

    def get_ttk_proxy(self, hero, out, feature_name):
        hp = _float((hero.get("actor_state") or {}).get("hp", 0.0))
        dps = []; self.get_dps_proxy(hero, dps, None)
        out.append(1e6 if dps[0] <= 1e-6 else hp / dps[0])

    def get_ttk_ratio(self, hero, out, feature_name):
        opp = self._enemy_of(hero)
        if not opp: out.append(1.0); return
        me = []; self.get_ttk_proxy(hero, me, None)
        he = []; self.get_ttk_proxy(opp, he, None)
        out.append(me[0] / max(1e-6, he[0]))

    # ================================================================== #
    # ========================= Tower risk trio ======================== #
    # ================================================================== #
    def _towers_by_camp(self, camp):
        towers = []
        for npc in self._frame_npcs:
            ast = npc or {}
            if ast.get("camp") != camp: continue
            sub = ast.get("sub_type") or ast.get("subType") or ""
            if sub in ("ACTOR_SUB_TOWER", "ACTOR_SUB_TOWER_SPRING", "ACTOR_SUB_CRYSTAL"):
                towers.append(ast)
        return towers

    def _nearest_tower_dist(self, hero, towers):
        if not towers: return float("inf"), None
        x, z = self._pos(hero)
        bestd = float("inf"); best = None
        for tw in towers:
            loc = (tw.get("location") or {})
            tx = _float(loc.get("x", 0.0)); tz = _float(loc.get("z", 0.0))
            if self.transform_camp2_to_camp1: tx, tz = -tx, -tz
            d = math.hypot(tx - x, tz - z)
            if d < bestd: bestd, best = d, tw
        return bestd, best

    def get_in_my_tower_range(self, hero, out, feature_name):
        my_camp = (hero.get("actor_state") or {}).get("camp")
        towers = self._towers_by_camp(my_camp)
        d, tw = self._nearest_tower_dist(hero, towers)
        if tw is None:
            out.append(0.0); return
        rng = _float(tw.get("attack_range", 0.0))
        out.append(1.0 if d <= rng else 0.0)

    def get_in_enemy_tower_range(self, hero, out, feature_name):
        my_camp = (hero.get("actor_state") or {}).get("camp")
        enemy_camp = "PLAYERCAMP_2" if my_camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        towers = self._towers_by_camp(enemy_camp)
        d, tw = self._nearest_tower_dist(hero, towers)
        if tw is None:
            out.append(0.0); return
        rng = _float(tw.get("attack_range", 0.0))
        out.append(1.0 if d <= rng else 0.0)

    def get_under_enemy_tower_fire(self, hero, out, feature_name):
        # best-effort: if bullets contain source runtime_id and it is a tower, mark 1
        hid = self._rid(hero)
        my_camp = (hero.get("actor_state") or {}).get("camp")
        enemy_camp = "PLAYERCAMP_2" if my_camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        enemy_towers = self._towers_by_camp(enemy_camp)
        tower_ids = { _int(tw.get("runtime_id", tw.get("config_id", -1)), -1) for tw in enemy_towers }

        firing = 0.0
        for b in self._frame_bullets:
            tgt = _int(b.get("target_actor", b.get("targetId", -1)), -1)
            if tgt != hid: continue
            src = _int(b.get("source_actor", b.get("sourceId", -1)), -1)
            if src in tower_ids:
                firing = 1.0; break
        out.append(firing)

    # ================================================================== #
    # ======================= Helpers / utilities ====================== #
    # ================================================================== #
    def _digits(self, feature_name: str) -> int:
        s = "".join([c for c in (feature_name or "") if c.isdigit()])
        return int(s or 0)
    
    def get_kill_income(self, hero, out, feature_name):
        """本命击杀收益（官方观测里常见的累积收益字段名：kill_income/killIncome）"""
        st = hero.get("actor_state", {}) or {}
        val = st.get("kill_income", st.get("killIncome", 0.0))
        out.append(_float(val, 0.0))  # NEW

    def get_money_cnt(self, hero, out, feature_name):
        """总经济（数据协议 HeroState.moneyCnt；兼容不同路径/别名）"""
        # 优先顶层 HeroState.moneyCnt；其次 actor_state.moneyCnt；再次 money 字段兜底
        top = hero.get("moneyCnt", None)
        if top is None:
            st = hero.get("actor_state", {}) or {}
            top = st.get("moneyCnt", st.get("money", hero.get("money", 0.0)))
        out.append(_float(top, 0.0))  # NEW
