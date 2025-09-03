#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Reward shaping for HoK 1v1 — Sun Shangxiang tailored (keep legacy API)
# - Public API unchanged:
#     GameRewardManager.__init__(main_hero_runtime_id)
#     GameRewardManager.result(frame_data) -> dict
#     GameRewardManager.frame_data_process(frame_data)
#     GameRewardManager.set_cur_calc_frame_vec(cul_calc_frame_map, frame_data, camp)
#     GameRewardManager.get_reward(frame_data, reward_dict)
#     GameRewardManager.calculate_forward(main_hero, main_tower, enemy_tower)
# - Weights are read ONLY from GameConfig.REWARD_WEIGHT_DICT (conf.py)
###########################################################################
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import math

try:
    from agent_ppo.conf.conf import GameConfig
except Exception:
    class GameConfig:
        REWARD_WEIGHT_DICT = {}
        TIME_SCALE_ARG = 0


# ------------------------------- utils ----------------------------------- #
@dataclass
class RewardStruct:
    weight: float
    value: float = 0.0
    cur_frame_value: float = 0.0
    last_frame_value: float = 0.0


def _safe_get(d: dict, path: List[str], default=None):
    x = d
    try:
        for k in path:
            if x is None:
                return default
            x = x.get(k)
        return default if x is None else x
    except Exception:
        return default


def _pos(actor_like: dict) -> Tuple[float, float]:
    ast = actor_like.get("actor_state", actor_like) or {}
    loc = ast.get("location") or {}
    return float(loc.get("x", 0.0)), float(loc.get("z", 0.0))


def _hp_ratio(hero: dict) -> float:
    ast = hero.get("actor_state") or {}
    hp = float(ast.get("hp", 0.0))
    mx = float(ast.get("max_hp", 1.0))
    return hp / mx if mx > 1e-6 else 0.0


def _hp_raw(hero: dict) -> float:
    return float(_safe_get(hero, ["actor_state", "hp"], 0.0))


def init_calc_frame_map() -> Dict[str, RewardStruct]:
    return {k: RewardStruct(weight=float(v)) for k, v in (GameConfig.REWARD_WEIGHT_DICT or {}).items()}


# ============================= Reward Manager ============================ #
class GameRewardManager:
    """
    与旧版保持一致的接口与策略：
      - DIFF_KEYS : 我方帧增量 − 敌方帧增量
      - EVENT_KEYS: 当前帧主客之差
      - ABS_KEYS  : 非零和，取我方值
    新增的孙尚香定制键（只有配权重才会生效）：
      skill2_hit_w, aa_on_s2_mark, s1_11110_use_chase, s1_11110_use_escape,
      enhanced_aa_after_s1, combo_2_1_aa, skill3_hit_w
    """
    # 旧有差分类
    DIFF_KEYS = {
        "hero_hp_point", "tower_hp_point", "gold_point", "ep_rate", "exp_point",
        "minion_push_depth",
    }
    # 旧有事件类 + 新事件类（技能/连招/普攻）
    EVENT_KEYS = {
        "kill_event", "death_event", "last_hit_event",
        "skill1_hit", "skill2_hit", "combo_21", "kill_finish_bonus",
        "skill2_hit_w", "aa_on_s2_mark", "s1_11110_use_chase", "s1_11110_use_escape",
        "enhanced_aa_after_s1", "combo_2_1_aa", "skill3_hit_w",
    }
    # 旧有绝对值类 + 更细致的行为
    ABS_KEYS = {
        "forward", "tower_danger", "dive_no_minion", "grass_engage",
        "hp_damage_adv", "low_hp_retreat", "chase_low_enemy",
    }

    LEVEL_MAX_EXP = {
        1: 160, 2: 298, 3: 446, 4: 524, 5: 613, 6: 713, 7: 825, 8: 950,
        9: 1088, 10: 1240, 11: 1406, 12: 1585, 13: 1778, 14: 1984, 15: 2200
    }

    # —— 技能ID（孙尚香）——
    SKILL_ID_S1 = 11110  # 翻滚突袭，赋能下一次普攻、近身加速
    SKILL_ID_S2 = 11120  # 红莲爆弹，命中标记降甲，普攻打标记有额外伤害
    SKILL_ID_S3 = 11130  # 究极弩炮，远程AOE

    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp: Optional[str] = None
        self.m_reward_value: Dict[str, float] = {}
        self.m_last_frame_no = -1
        self.time_scale_arg = getattr(GameConfig, "TIME_SCALE_ARG", 0)

        # 三套 calc map（与旧版兼容）
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}

        # 归一化：阵营总塔血（首帧记录）
        self._init_tower_total_hp = {"PLAYERCAMP_1": None, "PLAYERCAMP_2": None}

        # 事件 & 行为缓存
        self._last_kill_cnt = {"PLAYERCAMP_1": None, "PLAYERCAMP_2": None}
        self._last_dead_cnt = {"PLAYERCAMP_1": None, "PLAYERCAMP_2": None}
        self._last_hp_raw = {"PLAYERCAMP_1": None, "PLAYERCAMP_2": None}
        self._last_hp_rate = {"PLAYERCAMP_1": None, "PLAYERCAMP_2": None}
        self._last_dist = {"PLAYERCAMP_1": None, "PLAYERCAMP_2": None}
        self._last_pos = {"PLAYERCAMP_1": None, "PLAYERCAMP_2": None}

        # 技能槽计数
        self._last_slot_used  = {"PLAYERCAMP_1": [0, 0, 0, 0], "PLAYERCAMP_2": [0, 0, 0, 0]}
        self._last_slot_hit   = {"PLAYERCAMP_1": [0, 0, 0, 0], "PLAYERCAMP_2": [0, 0, 0, 0]}

        # —— 孙尚香专用窗口 —— #
        self._s1_last_use_frame = {"PLAYERCAMP_1": -10**9, "PLAYERCAMP_2": -10**9}
        self._s1_enhanced_pending = {"PLAYERCAMP_1": -1, "PLAYERCAMP_2": -1}      # S1 后等待强普命中
        self._s2_mark_active_until = {"PLAYERCAMP_1": -1, "PLAYERCAMP_2": -1}     # S2 命中后的普攻加成窗口
        self._combo_2_1_aa_until = {"PLAYERCAMP_1": -1, "PLAYERCAMP_2": -1}       # S2命中后→S1→强普 三连窗口

        self.m_each_level_max_exp = dict(self.LEVEL_MAX_EXP)
        self.RANGE_NORM = 15000.0  # 兵线/距离归一化兜底

    # ---------- 旧版公开 API ---------- #
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp = dict(self.LEVEL_MAX_EXP)

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data.get("frameNo", 0)
        if self.time_scale_arg and self.time_scale_arg > 0:
            scale = math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
            for key in list(self.m_reward_value.keys()):
                self.m_reward_value[key] *= scale
        return self.m_reward_value

    def frame_data_process(self, frame_data):
        # 识别我方阵营
        if self.main_hero_camp is None:
            target_id = str(self.main_hero_player_id)
            for hero in (frame_data.get("hero_states") or []):
                rid = str(_safe_get(hero, ["actor_state", "runtime_id"],
                                    _safe_get(hero, ["actor_state", "config_id"], "")))
                if rid and rid == target_id:
                    self.main_hero_camp = _safe_get(hero, ["actor_state", "camp"])
                    break
            if self.main_hero_camp is None and (frame_data.get("hero_states") or []):
                self.main_hero_camp = _safe_get(frame_data["hero_states"][0], ["actor_state", "camp"], "PLAYERCAMP_1")

        # 填充两侧“本帧原始值”
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, self.main_hero_camp)
        enemy_camp = "PLAYERCAMP_2" if self.main_hero_camp == "PLAYERCAMP_1" else "PLAYERCAMP_1"
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict: Dict[str, float]):
        reward_sum = 0.0
        for name, rs in self.m_main_calc_frame_map.items():
            w = rs.weight
            if name in self.DIFF_KEYS:
                cur_diff = self.m_main_calc_frame_map[name].cur_frame_value - self.m_enemy_calc_frame_map[name].cur_frame_value
                last_diff = self.m_main_calc_frame_map[name].last_frame_value - self.m_enemy_calc_frame_map[name].last_frame_value
                rs.value = cur_diff - last_diff
            elif name in self.EVENT_KEYS:
                rs.value = self.m_main_calc_frame_map[name].cur_frame_value - self.m_enemy_calc_frame_map[name].cur_frame_value
            else:  # ABS
                rs.value = self.m_main_calc_frame_map[name].cur_frame_value

            reward_sum += rs.value * w
            reward_dict[name] = rs.value

        # 滚动 last
        for m in (self.m_main_calc_frame_map, self.m_enemy_calc_frame_map):
            for k, v in m.items():
                v.last_frame_value = v.cur_frame_value

        reward_dict["reward_sum"] = reward_sum

    # 与旧版一致
    def set_cur_calc_frame_vec(self, cul_map, frame_data, camp):
        """按配置键填本阵营本帧值（不存在的键不计算）"""
        heroes = frame_data.get("hero_states") or []
        my = next((h for h in heroes if _safe_get(h, ["actor_state", "camp"]) == camp), None)
        enemy = next((h for h in heroes if _safe_get(h, ["actor_state", "camp"]) != camp), None)
        npcs = frame_data.get("npc_states") or []
        frame_no = int(frame_data.get("frameNo", 0))

        def _sub(n): return n.get("sub_type") or n.get("subType") or ""

        # 关键塔体
        def pick_tower(c):
            return next((n for n in npcs if (n.get("camp") == c and (_sub(n) in ("ACTOR_SUB_TOWER", "ACTOR_SUB_TOWER_SPRING")))), None)
        my_tower = pick_tower(camp)
        en_tower = pick_tower(self._camp_enemy(camp))

        # ---------- A) 敌我距离（用于多个项），首帧兜底 ----------
        dist = None; prev_dist = None
        if my and enemy:
            hx, hz = _pos(my); ex, ez = _pos(enemy)
            dist = math.hypot(hx - ex, hz - ez)
            prev_dist = self._last_dist.get(camp)
            if prev_dist is None or not isinstance(prev_dist, (int, float)) or not math.isfinite(prev_dist):
                prev_dist = dist  # 首帧兜底

        # ---------- B) 旧有基础：hp/ep/exp/gold ----------
        if "hero_hp_point" in cul_map:
            cul_map["hero_hp_point"].cur_frame_value = _hp_ratio(my) if my else 0.0
        if "ep_rate" in cul_map:
            vals = _safe_get(my or {}, ["actor_state", "values"], {})
            ep, mx = float(vals.get("ep", 0.0)), float(vals.get("max_ep", 1.0))
            cul_map["ep_rate"].cur_frame_value = (ep / mx) if mx > 1e-6 else 0.0
        if "exp_point" in cul_map:
            lvl = int((my or {}).get("level", 1)); cur = float((my or {}).get("exp", 0.0))
            mx = float(self.m_each_level_max_exp.get(max(1, min(15, lvl)), 1000.0))
            cul_map["exp_point"].cur_frame_value = max(0.0, min(1.0, cur / mx))
        if "gold_point" in cul_map:
            top = (my or {}).get("moneyCnt")
            if top is None:
                ast = (my or {}).get("actor_state", {}) or {}
                top = ast.get("moneyCnt", ast.get("gold", ast.get("money", 0.0)))
            cul_map["gold_point"].cur_frame_value = float(top or 0.0)

        # ---------- C) 阵营塔/水晶血量比 ----------
        if "tower_hp_point" in cul_map:
            tot_hp = 0.0
            for n in npcs:
                if n.get("camp") == camp and _sub(n) in ("ACTOR_SUB_TOWER", "ACTOR_SUB_TOWER_SPRING", "ACTOR_SUB_CRYSTAL"):
                    tot_hp += float(n.get("hp", 0.0))
            init = self._init_tower_total_hp.get(camp)
            if init is None or tot_hp > init:
                self._init_tower_total_hp[camp] = tot_hp
                init = tot_hp
            cul_map["tower_hp_point"].cur_frame_value = tot_hp / max(1e-6, init)

        # ---------- D) 兵线推进深度（差分） ----------
        if "minion_push_depth" in cul_map:
            def front_dist(owner: str, target_tower_camp: str) -> float:
                tw = pick_tower(target_tower_camp)
                if tw is None: return float("inf")
                tx, tz = _pos(tw); best = float("inf")
                for n in npcs:
                    if n.get("camp") != owner: continue
                    if "SOLDIER" not in _sub(n): continue
                    nx, nz = _pos(n); d = math.hypot(nx - tx, nz - tz)
                    if d < best: best = d
                return best
            my_to_enemy = front_dist(camp, self._camp_enemy(camp))
            his_to_my = front_dist(self._camp_enemy(camp), camp)
            if my_tower and en_tower:
                lane_len = max(1000.0, math.hypot(_pos(my_tower)[0] - _pos(en_tower)[0], _pos(my_tower)[1] - _pos(en_tower)[1]))
            else:
                lane_len = self.RANGE_NORM
            v_my = 1.0 - min(1.0, my_to_enemy / lane_len)
            v_his = 1.0 - min(1.0, his_to_my / lane_len)
            cul_map["minion_push_depth"].cur_frame_value = v_my - v_his

        # ---------- E) 前压（绝对值，塔下清零） ----------
        if "forward" in cul_map:
            fwd = 0.0
            if my and my_tower and en_tower:
                fwd = self.calculate_forward(my, my_tower, en_tower)
            if self._in_enemy_tower_range(npcs, my):
                fwd = 0.0
            cul_map["forward"].cur_frame_value = fwd

        # ---------- F) 塔域风险 ----------
        if "tower_danger" in cul_map or "dive_no_minion" in cul_map:
            in_range = 0.0; dive_no_minion = 0.0
            if my and en_tower:
                ex, ez = _pos(en_tower); hx, hz = _pos(my); atk_r = float(en_tower.get("attack_range", 0.0))
                in_range = 1.0 if (atk_r > 0 and math.hypot(hx - ex, hz - ez) <= atk_r) else 0.0
                near_cnt = 0
                if atk_r > 0:
                    for u in npcs:
                        if u.get("camp") != camp: continue
                        if "SOLDIER" not in _sub(u): continue
                        ux, uz = _pos(u)
                        if math.hypot(ux - ex, uz - ez) <= atk_r * 0.9:
                            near_cnt += 1
                dive_no_minion = 1.0 if (in_range and near_cnt == 0) else 0.0
            if "tower_danger" in cul_map:
                cul_map["tower_danger"].cur_frame_value = float(in_range)
            if "dive_no_minion" in cul_map:
                cul_map["dive_no_minion"].cur_frame_value = float(dive_no_minion)

        # ---------- G) 草丛贴脸 ----------
        if "grass_engage" in cul_map:
            val = 0.0
            if my and enemy:
                my_in = bool(my.get("isInGrass") or _safe_get(my, ["actor_state", "isInGrass"], False))
                his_in = bool(enemy.get("isInGrass") or _safe_get(enemy, ["actor_state", "isInGrass"], False))
                if my_in and not his_in:
                    hx, hz = _pos(my); ex, ez = _pos(enemy)
                    if math.hypot(hx - ex, hz - ez) <= 5000.0:
                        val = 0.1
            cul_map["grass_engage"].cur_frame_value = val

        # ---------- H) 敌我血量优势（绝对值）；同时维护 _last_hp_raw ----------
        if "hp_damage_adv" in cul_map and my and enemy:
            mycamp = camp; enemycamp = self._camp_enemy(camp)
            my_hp = _hp_raw(my); en_hp = _hp_raw(enemy)
            last_my = self._last_hp_raw.get(mycamp); last_en = self._last_hp_raw.get(enemycamp)
            dmg_to_en = max(0.0, (0.0 if last_en is None else last_en) - en_hp)
            dmg_to_me = max(0.0, (0.0 if last_my is None else last_my) - my_hp)
            cul_map["hp_damage_adv"].cur_frame_value = dmg_to_en - dmg_to_me
            self._last_hp_raw[mycamp] = my_hp
            self._last_hp_raw[enemycamp] = en_hp

        # ---------- I) 撤退/追击（绝对值） ----------
        if "low_hp_retreat" in cul_map and my and enemy and dist is not None:
            low_thr = 0.25
            retreating = (dist - prev_dist) > 50.0
            cul_map["low_hp_retreat"].cur_frame_value = 1.0 if (_hp_ratio(my) <= low_thr and retreating) else 0.0

        if "chase_low_enemy" in cul_map and my and enemy and dist is not None:
            low_thr_e = 0.25
            approaching = (prev_dist - dist) > 50.0
            safe = (not self._in_enemy_tower_range(npcs, my)) and (_hp_ratio(my) >= 0.30)
            cul_map["chase_low_enemy"].cur_frame_value = 1.0 if (_hp_ratio(enemy) <= low_thr_e and approaching and safe) else 0.0

        # ---------- J) 技能计数与ID ----------
        slots = (_safe_get(my, ["skill_state", "slot_states"], []) or [])
        def _field_list(f):
            out = []
            for s in slots: out.append(int(s.get(f, 0)))
            while len(out) < 4: out.append(0)
            return out
        used = _field_list("usedTimes")
        hit  = _field_list("hitHeroTimes")
        last_u = self._last_slot_used[camp]; last_h = self._last_slot_hit[camp]
        used_deltas = [max(0, used[i] - last_u[i]) for i in range(4)]
        hit_deltas  = [max(0,  hit[i] - last_h[i])  for i in range(4)]

        # 尝试读取 skill_id（没有就 -1）
        slot_ids = []
        for s in slots:
            sid = s.get("skill_id", s.get("skillId", s.get("config_id", s.get("id", None))))
            slot_ids.append(int(sid) if sid is not None else -1)
        while len(slot_ids) < 4:
            slot_ids.append(-1)

        # 维护 last
        self._last_slot_used[camp] = used
        self._last_slot_hit[camp]  = hit

        # ---------- K) 旧有：简单命中统计（兼容） ----------
        if "skill1_hit" in cul_map:
            cul_map["skill1_hit"].cur_frame_value = float(hit_deltas[1])
        if "skill2_hit" in cul_map:
            cul_map["skill2_hit"].cur_frame_value = float(hit_deltas[2])
        if "combo_21" in cul_map:
            # 旧逻辑：S2→S1 命中（简版），这里保留但不强依赖
            val = 1.0 if (used_deltas[2] > 0 and hit_deltas[1] > 0) else 0.0
            cul_map["combo_21"].cur_frame_value = val

        # ---------- L) S2（11120）命中质量 & 标记窗口 ----------
        s2_is_11120 = (len(slot_ids) > 2) and (slot_ids[2] in (self.SKILL_ID_S2, -1))  # 无id则兜底视为该英雄
        if "skill2_hit_w" in cul_map:
            s2_hits = float(hit_deltas[2])
            weight = 1.0
            if my and enemy:
                # 更近更值钱，塔下打折；命中为主
                if dist is not None:
                    weight *= max(0.6, 1.2 - min(1.0, dist / 9000.0))
                if self._in_enemy_tower_range(npcs, my):
                    weight *= 0.7
            cul_map["skill2_hit_w"].cur_frame_value = s2_hits * weight

        # S2 命中→开启“被标记”窗口（近似 120 帧）
        if s2_is_11120 and hit_deltas[2] > 0:
            self._s2_mark_active_until[camp] = frame_no + 120  # ~2s，可按实际调

        # ---------- M) S1（11110）释放质量（追击/撤退） ----------
        s1_is_11110 = (len(slot_ids) > 1) and (slot_ids[1] in (self.SKILL_ID_S1, -1))
        if (used_deltas[1] > 0) and s1_is_11110 and my and enemy and dist is not None:
            last_pos = self._last_pos.get(camp)
            move_towards = move_away = False
            if last_pos is not None:
                mx, mz = _pos(my)[0] - last_pos[0], _pos(my)[1] - last_pos[1]
                vx, vz = _pos(enemy)[0] - last_pos[0], _pos(enemy)[1] - last_pos[1]
                dot = mx * vx + mz * vz
                if dot > 0: move_towards = True
                if dot < 0: move_away = True
            my_hp = _hp_ratio(my); en_hp = _hp_ratio(enemy)

            if "s1_11110_use_chase" in cul_map:
                ok = (en_hp <= 0.5 or move_towards) and (not self._in_enemy_tower_range(npcs, my)) and (my_hp >= 0.30)
                cul_map["s1_11110_use_chase"].cur_frame_value = 1.0 if ok else 0.0

            if "s1_11110_use_escape" in cul_map:
                ok = (my_hp <= 0.30) and move_away
                cul_map["s1_11110_use_escape"].cur_frame_value = 1.0 if ok else 0.0

            # S1 后开启“强化普攻”窗口（60 帧）
            self._s1_last_use_frame[camp] = frame_no
            self._s1_enhanced_pending[camp] = frame_no + 60

            # 若 S2 刚命中过（≤20 帧），则开启 “S2→S1→强普” 的三连窗口（再等 20 帧）
            if frame_no <= self._s2_mark_active_until.get(camp, -1) and (frame_no - (self._s1_last_use_frame[camp])) <= 1:
                self._combo_2_1_aa_until[camp] = frame_no + 20

        # ---------- N) 强化普攻命中（S1后） & 三连达成 ----------
        def _basic_attack_hit_this_frame() -> bool:
            # 启发式：若本帧没有技能命中（sum(hit_deltas)==0），但敌方hp下降超过阈值，则视为普攻命中
            enemycamp = self._camp_enemy(camp)
            last_en = self._last_hp_raw.get(enemycamp)
            if last_en is None or enemy is None:
                return False
            en_hp_now = _hp_raw(enemy)
            dmg = max(0.0, last_en - en_hp_now)
            no_skill_hit_now = (sum(hit_deltas) == 0)
            # 阈值按你日志量级微调
            ratio_drop = 0.0
            last_en_ratio = self._last_hp_rate.get(enemycamp, None)
            if last_en_ratio is not None and last_en > 1e-6:
                ratio_drop = max(0.0, (last_en - en_hp_now) / last_en)
            return no_skill_hit_now and (dmg >= 50.0 or ratio_drop >= 0.01)

        enhanced_hit = _basic_attack_hit_this_frame()

        if "enhanced_aa_after_s1" in cul_map:
            val = 0.0
            if enhanced_hit and frame_no <= self._s1_enhanced_pending.get(camp, -1):
                val = 1.0
                self._s1_enhanced_pending[camp] = -1
            cul_map["enhanced_aa_after_s1"].cur_frame_value = val

        if "aa_on_s2_mark" in cul_map:
            val = 0.0
            if enhanced_hit and frame_no <= self._s2_mark_active_until.get(camp, -1):
                val = 1.0
            cul_map["aa_on_s2_mark"].cur_frame_value = val

        if "combo_2_1_aa" in cul_map:
            val = 0.0
            if enhanced_hit and frame_no <= self._combo_2_1_aa_until.get(camp, -1):
                val = 1.0
                self._combo_2_1_aa_until[camp] = -1
                self._s1_enhanced_pending[camp] = -1
            cul_map["combo_2_1_aa"].cur_frame_value = val

        # ---------- O) 大招（11130）命中质量 ----------
        s3_is_11130 = (len(slot_ids) > 3) and (slot_ids[3] in (self.SKILL_ID_S3, -1))
        if "skill3_hit_w" in cul_map:
            s3_hits = float(hit_deltas[3])
            weight = 1.0
            if my and enemy and dist is not None:
                # 远距离命中更难，给轻微加权；塔下折扣
                weight *= (1.0 + min(0.5, dist / 12000.0))  # up to ×1.5
                if self._in_enemy_tower_range(npcs, my):
                    weight *= 0.8
            cul_map["skill3_hit_w"].cur_frame_value = s3_hits * weight

        # ---------- P) 击杀/死亡/补刀（事件） ----------
        if ("kill_event" in cul_map) or ("death_event" in cul_map):
            k_ev, d_ev = self._kill_death_event(frame_data, camp, my)
            if "kill_event" in cul_map:  cul_map["kill_event"].cur_frame_value  = k_ev
            if "death_event" in cul_map: cul_map["death_event"].cur_frame_value = d_ev
        if "last_hit_event" in cul_map:
            cul_map["last_hit_event"].cur_frame_value = self._last_hit_event(frame_data, camp)

        # ---------- Q) 维护“上一帧距离/位置/血量比” ----------
        if my is not None:
            self._last_pos[camp] = _pos(my)
            self._last_dist[camp] = dist if dist is not None else self._last_dist.get(camp)
            self._last_hp_rate[camp] = _hp_ratio(my)

    # 与旧版一致：前压 = 1 - d(hero, enemy_tower)/d(my_tower, enemy_tower) 乘 hp 比
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = _pos(main_tower)
        enemy_tower_pos = _pos(enemy_tower)
        hero_pos = _pos(main_hero)
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = max(math.dist(main_tower_pos, enemy_tower_pos), 1e-6)
        base = (dist_main2emy - dist_hero2emy) / dist_main2emy
        base = max(0.0, base)
        return base * _hp_ratio(main_hero)

    # ------------------------------ helpers ------------------------------ #
    def _camp_enemy(self, c):
        return "PLAYERCAMP_2" if c == "PLAYERCAMP_1" else "PLAYERCAMP_1"

    def _in_enemy_tower_range(self, npc_list: List[dict], hero: dict) -> bool:
        if not hero:
            return False
        hx, hz = _pos(hero)
        mycamp = _safe_get(hero, ["actor_state", "camp"])
        enemy_camp = self._camp_enemy(mycamp)
        for n in npc_list or []:
            if n.get("camp") != enemy_camp: continue
            sub = n.get("sub_type") or n.get("subType") or ""
            if sub not in ("ACTOR_SUB_TOWER", "ACTOR_SUB_TOWER_SPRING"): continue
            tx, tz = _pos(n); rng = float(n.get("attack_range", 0.0))
            if rng > 0 and math.hypot(tx - hx, tz - hz) <= rng:
                return True
        return False

    def _kill_death_event(self, frame_data: dict, camp: str, hero: dict) -> Tuple[float, float]:
        k_ev = d_ev = 0.0
        # A) frame_action 优先
        for a in (frame_data.get("frame_action") or []):
            da = a.get("dead_action") or {}
            killer = _safe_get(da, ["killer", "camp"])
            victim = _safe_get(da, ["death", "camp"])
            if killer == camp and victim == self._camp_enemy(camp): k_ev += 1.0
            if victim == camp: d_ev += 1.0
        # B) 退化：计数器差分
        kill_cnt = int((hero or {}).get("killCnt", 0))
        dead_cnt = int((hero or {}).get("deadCnt", 0))
        last_k = self._last_kill_cnt.get(camp)
        last_d = self._last_dead_cnt.get(camp)
        if last_k is not None: k_ev += max(0, kill_cnt - last_k)
        if last_d is not None: d_ev += max(0, dead_cnt - last_d)
        self._last_kill_cnt[camp] = kill_cnt
        self._last_dead_cnt[camp] = dead_cnt
        return k_ev, d_ev

    def _last_hit_event(self, frame_data: dict, camp: str) -> float:
        cnt = 0.0
        for a in (frame_data.get("frame_action") or []):
            da = a.get("dead_action") or {}
            death = da.get("death") or {}
            if str(death.get("type", "")).startswith("SOLDIER") and _safe_get(da, ["killer", "camp"]) == camp:
                cnt += 1.0
        return cnt
