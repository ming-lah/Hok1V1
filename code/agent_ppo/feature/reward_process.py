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
from collections import deque

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
class SunShangxiangComboDetector:
    """孙尚香连招检测器 - 基于论文洞察的精确连招识别"""
    
    def __init__(self):
        self.skill_history = deque(maxlen=20)
        self.combo_states = {
            's2_mark_active_until': -1,      # S2标记激活截止帧
            's1_enhanced_pending': -1,       # S1强化普攻待执行截止帧
            'combo_2_1_aa_until': -1,        # S2→S1→强普连招窗口截止帧
        }
        
    def update_skill_usage(self, hero_state: dict, frame_no: int):
        """更新技能使用记录"""
        skill_state = hero_state.get('skill_state', {})
        slots = skill_state.get('slot_states', [])
        
        for i, slot in enumerate(slots[:4]):
            used_times = int(slot.get('usedTimes', 0))
            hit_times = int(slot.get('hitHeroTimes', 0))
            
            # 检测新的技能使用
            if len(self.skill_history) == 0 or used_times > self.skill_history[-1].get(f'slot_{i}_used', 0):
                skill_type = self._get_skill_type_by_slot(i, slot.get('configId', 0))
                self.skill_history.append({
                    'frame_no': frame_no,
                    'skill_type': skill_type,
                    'slot_index': i,
                    'hit_hero': hit_times > self.skill_history[-1].get(f'slot_{i}_hit', 0) if self.skill_history else False,
                    f'slot_{i}_used': used_times,
                    f'slot_{i}_hit': hit_times
                })
                
                # 更新连招状态
                self._update_combo_states(skill_type, frame_no)
    
    def _get_skill_type_by_slot(self, slot_index: int, config_id: int) -> str:
        """根据技能槽和配置ID确定技能类型"""
        if slot_index == 1 and config_id == 11110:
            return 'skill1_tumble'
        elif slot_index == 2 and config_id == 11120:
            return 'skill2_red_lotus'
        elif slot_index == 3 and config_id == 11130:
            return 'skill3_ultimate'
        elif slot_index == 0:
            return 'normal_attack'
        else:
            return f'unknown_skill_{slot_index}'
    
    def _update_combo_states(self, skill_type: str, frame_no: int):
        """更新连招状态窗口"""
        if skill_type == 'skill2_red_lotus':
            # S2命中后开启标记窗口（2秒 = 120帧）
            self.combo_states['s2_mark_active_until'] = frame_no + 120
            
        elif skill_type == 'skill1_tumble':
            # S1使用后开启强化普攻窗口（1秒 = 60帧）
            self.combo_states['s1_enhanced_pending'] = frame_no + 60
            
            # 如果在S2标记窗口内使用S1，开启三连窗口
            if frame_no <= self.combo_states['s2_mark_active_until']:
                self.combo_states['combo_2_1_aa_until'] = frame_no + 60
    
    def detect_combo_execution(self, frame_no: int) -> tuple:
        """检测连招执行情况"""
        combo_type = None
        reward_multiplier = 1.0
        
        # 检测S2→S1→强普三连
        if (frame_no <= self.combo_states['combo_2_1_aa_until'] and 
            self._has_recent_enhanced_attack(frame_no)):
            combo_type = 's2_s1_enhanced_aa'
            reward_multiplier = 3.0
            # 重置窗口
            self.combo_states['combo_2_1_aa_until'] = -1
            self.combo_states['s1_enhanced_pending'] = -1
            
        # 检测S1→强普连招
        elif (frame_no <= self.combo_states['s1_enhanced_pending'] and 
              self._has_recent_enhanced_attack(frame_no)):
            combo_type = 's1_enhanced_aa'
            reward_multiplier = 1.8
            self.combo_states['s1_enhanced_pending'] = -1
        
        # 检测S2标记期间普攻
        elif (frame_no <= self.combo_states['s2_mark_active_until'] and 
              self._has_recent_normal_attack(frame_no)):
            combo_type = 's2_mark_aa'
            reward_multiplier = 1.5
        
        return combo_type, reward_multiplier
    
    def _has_recent_enhanced_attack(self, frame_no: int, window: int = 10) -> bool:
        """检测最近是否有强化普攻"""
        for record in reversed(list(self.skill_history)[-5:]):
            if (frame_no - record['frame_no'] <= window and 
                record['skill_type'] == 'normal_attack' and 
                record['hit_hero']):
                return True
        return False
    
    def _has_recent_normal_attack(self, frame_no: int, window: int = 10) -> bool:
        """检测最近是否有普攻"""
        for record in reversed(list(self.skill_history)[-3:]):
            if (frame_no - record['frame_no'] <= window and 
                record['skill_type'] == 'normal_attack'):
                return True
        return False


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

        # —— 孙尚香专用增强系统 —— #
        self.sunshangxiang_combo_detector = SunShangxiangComboDetector()
        self.marksman_tactical_analyzer = MarksmanTacticalAnalyzer()
        self.adaptive_reward_weights = AdaptiveRewardWeights()
        self.economic_game_analyzer = EconomicGameAnalyzer()  # 新增经济博弈分析器
        
        # 新增：稠密奖励处理器 (基于深度RL分析)
        from agent_ppo.feature.dense_reward_processor import DenseRewardProcessor
        self.dense_reward_processor = DenseRewardProcessor(main_hero_runtime_id, self.main_hero_camp)
        
        # 新增：防御塔战略奖励处理器 (基于防御塔核心分析)
        from agent_ppo.feature.tower_strategic_rewards import TowerStrategicRewards
        self.tower_strategic_rewards = TowerStrategicRewards(main_hero_runtime_id, self.main_hero_camp)
        
        # 新增：兵线战略奖励处理器 (基于兵线战略分析)
        from agent_ppo.feature.minion_strategic_rewards import MinionStrategicRewards
        self.minion_strategic_rewards = MinionStrategicRewards(main_hero_runtime_id, self.main_hero_camp)
        
        # 新增：草丛战术奖励处理器 (基于草丛战术分析)
        from agent_ppo.feature.bush_tactical_rewards import BushTacticalRewards
        self.bush_tactical_rewards = BushTacticalRewards(main_hero_runtime_id, self.main_hero_camp)
        
        # 原有孙尚香窗口（保持兼容）
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
            hero_states = frame_data.get("hero_states") or []
            if isinstance(hero_states, list):
                for hero in hero_states:
                    if not isinstance(hero, dict):
                        continue
                    
                    rid = str(_safe_get(hero, ["actor_state", "runtime_id"],
                                        _safe_get(hero, ["actor_state", "config_id"], "")))
                    if rid and rid == target_id:
                        self.main_hero_camp = _safe_get(hero, ["actor_state", "camp"])
                        break
                        
                if self.main_hero_camp is None and hero_states:
                    first_hero = hero_states[0]
                    if isinstance(first_hero, dict):
                        self.main_hero_camp = _safe_get(first_hero, ["actor_state", "camp"], "PLAYERCAMP_1")

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
        frame_actions = frame_data.get("frame_action") or []
        if isinstance(frame_actions, list):
            for a in frame_actions:
                # 添加类型检查
                if not isinstance(a, dict):
                    continue
                
                da = a.get("dead_action") or {}
                if not isinstance(da, dict):
                    continue
                
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
        frame_actions = frame_data.get("frame_action") or []
        if isinstance(frame_actions, list):
            for a in frame_actions:
                if not isinstance(a, dict):
                    continue
                
                da = a.get("dead_action") or {}
                if not isinstance(da, dict):
                    continue
                
                death = da.get("death") or {}
                if not isinstance(death, dict):
                    continue
                
                if str(death.get("type", "")).startswith("SOLDIER") and _safe_get(da, ["killer", "camp"]) == camp:
                    cnt += 1.0
        return cnt
    
    def calculate_economic_game_rewards(self, frame_data: dict, main_hero: dict, 
                                      enemy_hero: dict, frame_no: int) -> Dict[str, float]:
        """
        计算经济博弈奖励 - 基于用户零和博弈思路的实现
        
        核心逻辑：
        1. 计算经济差值delta = 我方经济 - 敌方经济
        2. 如果delta > 0 且敌方在攻击范围内 → 优先攻击英雄
        3. 如果delta > 0 但敌方不在攻击范围内 → 转向攻击小兵获得经济
        4. 如果delta < 0 → 专注发育追赶经济
        """
        rewards = {}
        
        if not main_hero or not enemy_hero:
            return rewards
        
        # 获取经济数据
        my_money = float(main_hero.get("money", 0))
        enemy_money = float(enemy_hero.get("money", 0))
        money_delta = my_money - enemy_money
        
        # 获取位置和距离信息
        my_pos = _pos(main_hero)
        enemy_pos = _pos(enemy_hero)
        distance = math.hypot(my_pos[0] - enemy_pos[0], my_pos[1] - enemy_pos[1])
        actor_state = main_hero.get("actor_state", {})
        if isinstance(actor_state, dict):
            my_attack_range = float(actor_state.get("attack_range", 600))
        else:
            my_attack_range = 600.0
        
        # 核心零和博弈逻辑奖励
        if money_delta > 0:
            # 经济优势状态
            if distance <= my_attack_range:
                # 在攻击范围内，应该优先攻击英雄
                hero_attack_reward = self._calculate_hero_attack_reward(
                    main_hero, enemy_hero, frame_data, money_delta
                )
                rewards["economic_hero_priority"] = hero_attack_reward
                
                # 如果攻击小兵而不是英雄，给予惩罚
                minion_attack_penalty = self._calculate_minion_attack_penalty(
                    frame_data, money_delta
                )
                rewards["economic_minion_penalty"] = minion_attack_penalty
                
            else:
                # 不在攻击范围内，应该转向攻击小兵
                farming_reward = self._calculate_economic_farming_reward(
                    main_hero, frame_data, money_delta
                )
                rewards["economic_farming_priority"] = farming_reward
                
                # 追击奖励（拉近距离以便攻击英雄）
                pursuit_reward = self._calculate_economic_pursuit_reward(
                    my_pos, enemy_pos, distance, my_attack_range
                )
                rewards["economic_pursuit"] = pursuit_reward
        
        elif money_delta < -500:  # 经济劣势超过500金
            # 经济劣势状态，专注发育
            catchup_farming_reward = self._calculate_catchup_farming_reward(
                main_hero, frame_data, abs(money_delta)
            )
            rewards["economic_catchup_farming"] = catchup_farming_reward
            
            # 避战奖励（保守发育）
            safe_farming_reward = self._calculate_safe_farming_reward(
                main_hero, enemy_hero, frame_data, distance
            )
            rewards["economic_safe_farming"] = safe_farming_reward
        
        # 经济效率奖励
        economic_efficiency_reward = self._calculate_economic_efficiency_reward(
            main_hero, frame_data, frame_no
        )
        rewards["economic_efficiency"] = economic_efficiency_reward
        
        # 装备时机奖励
        equipment_timing_reward = self._calculate_equipment_timing_reward(
            main_hero, my_money
        )
        rewards["economic_equipment_timing"] = equipment_timing_reward
        
        return rewards
    
    def _calculate_hero_attack_reward(self, main_hero: dict, enemy_hero: dict, 
                                    frame_data: dict, money_delta: float) -> float:
        """计算攻击英雄的奖励（经济优势时）"""
        # 检查是否对英雄造成伤害
        hero_damage = 0.0
        hero_states = frame_data.get("hero_states", [])
        if isinstance(hero_states, list):
            for hero_state in hero_states:
                if not isinstance(hero_state, dict):
                    continue
                
                if hero_state.get("player_id") == main_hero.get("player_id"):
                    # 检查伤害输出
                    take_hurt_infos = hero_state.get("takeHurtInfos", [])
                    if isinstance(take_hurt_infos, list):
                        for hurt_info in take_hurt_infos:
                            if isinstance(hurt_info, dict) and hurt_info.get("source_actor") == enemy_hero.get("player_id"):
                                hero_damage += float(hurt_info.get("damage", 0))
        
        # 基础奖励：经济优势越大，攻击英雄奖励越高
        base_reward = min(money_delta / 2000.0, 2.0)  # 最高2.0奖励
        
        # 伤害奖励
        damage_reward = min(hero_damage / 100.0, 1.0)  # 每100伤害给1.0奖励
        
        return base_reward + damage_reward
    
    def _calculate_minion_attack_penalty(self, frame_data: dict, money_delta: float) -> float:
        """计算攻击小兵的惩罚（经济优势且敌人在攻击范围内时）"""
        # 检查是否击杀了小兵而不是攻击英雄
        minion_kills = 0
        frame_actions = frame_data.get("frame_action", [])
        if isinstance(frame_actions, list):
            for action in frame_actions:
                if not isinstance(action, dict):
                    continue
                
                dead_action = action.get("dead_action", {})
                if not isinstance(dead_action, dict):
                    continue
                
                death = dead_action.get("death", {})
                if isinstance(death, dict) and "SOLDIER" in str(death.get("type", "")):
                    minion_kills += 1
        
        if minion_kills > 0:
            # 经济优势越大，错失攻击英雄机会的惩罚越大
            penalty_multiplier = min(money_delta / 1000.0, 1.5)
            return -0.5 * penalty_multiplier * minion_kills
        
        return 0.0
    
    def _calculate_economic_farming_reward(self, main_hero: dict, frame_data: dict, 
                                         money_delta: float) -> float:
        """计算经济发育奖励（经济优势但敌人不在攻击范围内时）"""
        # 补刀奖励
        last_hit_reward = 0.0
        frame_actions = frame_data.get("frame_action", [])
        if isinstance(frame_actions, list):
            for action in frame_actions:
                if not isinstance(action, dict):
                    continue
                
                dead_action = action.get("dead_action", {})
                if not isinstance(dead_action, dict):
                    continue
                
                death = dead_action.get("death", {})
                killer = dead_action.get("killer", {})
                
                if (isinstance(death, dict) and isinstance(killer, dict) and
                    "SOLDIER" in str(death.get("type", "")) and 
                    killer.get("player_id") == main_hero.get("player_id")):
                    last_hit_reward += 0.8  # 每个补刀0.8奖励
        
        # 经济优势时的发育效率加成
        efficiency_multiplier = 1.0 + min(money_delta / 3000.0, 0.5)
        
        return last_hit_reward * efficiency_multiplier
    
    def _calculate_economic_pursuit_reward(self, my_pos: tuple, enemy_pos: tuple, 
                                         current_distance: float, attack_range: float) -> float:
        """计算经济优势时的追击奖励"""
        # 如果距离在合理追击范围内（攻击范围的1.5倍内）
        pursuit_range = attack_range * 1.5
        
        if current_distance <= pursuit_range:
            # 距离越接近攻击范围，奖励越高
            distance_ratio = (pursuit_range - current_distance) / pursuit_range
            return 0.3 * distance_ratio
        
        return 0.0
    
    def _calculate_catchup_farming_reward(self, main_hero: dict, frame_data: dict, 
                                        money_deficit: float) -> float:
        """计算追赶经济的发育奖励"""
        # 补刀奖励（经济劣势时更重要）
        last_hit_reward = 0.0
        frame_actions = frame_data.get("frame_action", [])
        if isinstance(frame_actions, list):
            for action in frame_actions:
                if not isinstance(action, dict):
                    continue
                
                dead_action = action.get("dead_action", {})
                if not isinstance(dead_action, dict):
                    continue
                
                death = dead_action.get("death", {})
                killer = dead_action.get("killer", {})
                
                if (isinstance(death, dict) and isinstance(killer, dict) and
                    "SOLDIER" in str(death.get("type", "")) and 
                    killer.get("player_id") == main_hero.get("player_id")):
                    last_hit_reward += 1.2  # 经济劣势时补刀奖励更高
        
        # 劣势越大，发育奖励越高
        deficit_multiplier = 1.0 + min(money_deficit / 2000.0, 1.0)
        
        return last_hit_reward * deficit_multiplier
    
    def _calculate_safe_farming_reward(self, main_hero: dict, enemy_hero: dict, 
                                     frame_data: dict, distance: float) -> float:
        """计算安全发育奖励"""
        my_hp_ratio = _hp_ratio(main_hero)
        enemy_hp_ratio = _hp_ratio(enemy_hero)
        
        # 安全距离奖励（血量劣势时保持距离）
        safe_distance_reward = 0.0
        if my_hp_ratio < enemy_hp_ratio and distance > 1000:
            safe_distance_reward = 0.4
        
        # 发育时的血量管理奖励
        hp_management_reward = 0.0
        if my_hp_ratio > 0.7:  # 保持健康血量
            hp_management_reward = 0.2
        
        return safe_distance_reward + hp_management_reward
    
    def _calculate_economic_efficiency_reward(self, main_hero: dict, frame_data: dict, 
                                            frame_no: int) -> float:
        """计算经济效率奖励"""
        current_money = float(main_hero.get("money", 0))
        total_money = float(main_hero.get("moneyCnt", 0))
        
        # 计算每分钟经济
        game_minutes = max(frame_no / 1800.0, 1.0)  # 30fps * 60s = 1800 frames/min
        money_per_minute = total_money / game_minutes
        
        # 经济效率评分
        if money_per_minute > 1500:  # 优秀经济效率
            return 0.5
        elif money_per_minute > 1200:  # 良好经济效率
            return 0.3
        elif money_per_minute > 900:  # 一般经济效率
            return 0.1
        else:
            return -0.1  # 经济效率过低，给予轻微惩罚
    
    def _calculate_equipment_timing_reward(self, main_hero: dict, current_money: float) -> float:
        """计算装备购买时机奖励"""
        # 检查装备状态变化（简化实现）
        # 在实际实现中，需要对比前后帧的装备状态
        
        # 基于当前金币数量的装备建议
        if 2800 <= current_money <= 3200:  # 适合买大件的金币数
            return 0.3
        elif 1400 <= current_money <= 1600:  # 适合买中等装备
            return 0.2
        elif 700 <= current_money <= 900:  # 适合买小件
            return 0.1
        elif current_money > 4000:  # 金币过多未及时购买装备
            return -0.2
        
        return 0.0
    
    def calculate_comprehensive_rewards(self, frame_data: dict, main_hero: dict, 
                                      enemy_hero: dict, frame_no: int) -> Dict[str, float]:
        """
        计算全面的稠密奖励 - 基于深度RL分析的完整奖励系统
        
        整合了：
        1. 原有的经济博弈奖励 (零和博弈思路)
        2. 新的稠密奖励系统 (完整的奖励塑形)
        3. 动态权重调整
        """
        all_rewards = {}
        
        # 1. 原有的经济博弈奖励
        economic_rewards = self.calculate_economic_game_rewards(frame_data, main_hero, enemy_hero, frame_no)
        all_rewards.update(economic_rewards)
        
        # 2. 稠密奖励系统
        dense_rewards = self.dense_reward_processor.calculate_dense_rewards(
            frame_data, main_hero, enemy_hero, frame_no
        )
        all_rewards.update(dense_rewards)
        
        # 3. 防御塔战略奖励系统 (基于您的防御塔核心分析)
        tower_strategic_rewards = self.tower_strategic_rewards.calculate_tower_strategic_rewards(
            frame_data, main_hero, enemy_hero, frame_no
        )
        all_rewards.update(tower_strategic_rewards)
        
        # 4. 兵线战略奖励系统 (基于您的兵线战略分析)
        minion_strategic_rewards = self.minion_strategic_rewards.calculate_minion_strategic_rewards(
            frame_data, main_hero, enemy_hero, frame_no
        )
        all_rewards.update(minion_strategic_rewards)
        
        # 5. 草丛战术奖励系统 (基于您的草丛战术分析)
        bush_tactical_rewards = self.bush_tactical_rewards.calculate_bush_tactical_rewards(
            frame_data, main_hero, enemy_hero, frame_no
        )
        all_rewards.update(bush_tactical_rewards)
        
        # 6. 应用自适应权重
        adaptive_weights = self.adaptive_reward_weights.get_current_weights()
        
        # 7. 权重调整和奖励融合
        final_rewards = {}
        for reward_type, reward_value in all_rewards.items():
            # 应用配置中的权重
            if reward_type in GameConfig.REWARD_WEIGHT_DICT:
                base_weight = GameConfig.REWARD_WEIGHT_DICT[reward_type]
            else:
                base_weight = 1.0
            
            # 应用自适应权重
            adaptive_multiplier = 1.0
            for adaptive_key, adaptive_value in adaptive_weights.items():
                if adaptive_key.replace('_reward_weight', '') in reward_type:
                    adaptive_multiplier = adaptive_value
                    break
            
            final_reward = reward_value * base_weight * adaptive_multiplier
            final_rewards[reward_type] = final_reward
        
        return final_rewards


class MarksmanTacticalAnalyzer:
    """射手战术分析器 - 基于论文2的射手专用战术评估"""
    
    def __init__(self):
        self.position_history = deque(maxlen=10)
        self.damage_windows = {}
        
    def analyze_kiting_execution(self, my_hero: dict, enemy_hero: dict, frame_no: int) -> float:
        """分析风筝战术执行质量"""
        if not my_hero or not enemy_hero:
            return 0.0
            
        # 获取位置和距离
        my_pos = _pos(my_hero)
        enemy_pos = _pos(enemy_hero)
        current_distance = math.hypot(my_pos[0] - enemy_pos[0], my_pos[1] - enemy_pos[1])
        
        # 记录位置历史
        self.position_history.append({
            'frame_no': frame_no,
            'my_pos': my_pos,
            'enemy_pos': enemy_pos,
            'distance': current_distance
        })
        
        if len(self.position_history) < 3:
            return 0.0
            
        # 分析移动趋势
        recent_positions = list(self.position_history)[-3:]
        distance_changes = []
        
        for i in range(1, len(recent_positions)):
            distance_change = recent_positions[i]['distance'] - recent_positions[i-1]['distance']
            distance_changes.append(distance_change)
        
        # 评估风筝质量
        avg_distance_change = sum(distance_changes) / len(distance_changes) if distance_changes else 0
        
        # 获取攻击范围
        actor_state = my_hero.get("actor_state", {})
        if isinstance(actor_state, dict):
            my_range = float(actor_state.get("attack_range", 600))
        else:
            my_range = 600.0
        optimal_distance = my_range * 0.85  # 理想距离为攻击范围的85%
        
        # 风筝质量评分
        if current_distance > optimal_distance and avg_distance_change > 0:
            # 保持安全距离且在拉开
            return min(1.0, avg_distance_change / 200.0)
        elif current_distance < optimal_distance * 0.7:
            # 距离过近，惩罚
            return -0.3
        else:
            return 0.0
    
    def analyze_positioning_quality(self, my_hero: dict, enemy_hero: dict, frame_state: dict) -> float:
        """分析位置质量"""
        if not my_hero or not enemy_hero:
            return 0.0
            
        my_pos = _pos(my_hero)
        enemy_pos = _pos(enemy_hero)
        
        # 分析与防御塔的关系
        tower_score = self._analyze_tower_positioning(my_pos, enemy_pos, frame_state)
        
        # 分析与小兵的关系
        minion_score = self._analyze_minion_positioning(my_pos, frame_state)
        
        # 分析攻击角度
        angle_score = self._analyze_attack_angle(my_hero, enemy_hero)
        
        return (tower_score + minion_score + angle_score) / 3.0
    
    def _analyze_tower_positioning(self, my_pos: tuple, enemy_pos: tuple, frame_state: dict) -> float:
        """分析塔位关系"""
        npcs = frame_state.get('npc_states', [])
        
        # 找到敌方防御塔
        enemy_towers = []
        for npc in npcs:
            if 'TOWER' in npc.get('sub_type', '') and npc.get('camp') != self._get_my_camp():
                tower_pos = npc.get('location', {})
                if tower_pos:
                    enemy_towers.append((float(tower_pos.get('x', 0)), float(tower_pos.get('z', 0))))
        
        if not enemy_towers:
            return 0.0
            
        # 计算与最近敌塔的距离
        min_tower_dist = float('inf')
        for tower_pos in enemy_towers:
            dist = math.hypot(my_pos[0] - tower_pos[0], my_pos[1] - tower_pos[1])
            min_tower_dist = min(min_tower_dist, dist)
        
        # 安全距离评分
        if min_tower_dist > 1200:  # 安全距离
            return 0.8
        elif min_tower_dist > 800:  # 边缘距离
            return 0.3
        else:  # 危险距离
            return -1.0
    
    def _analyze_minion_positioning(self, my_pos: tuple, frame_state: dict) -> float:
        """分析与小兵的位置关系"""
        # 简化实现
        return 0.0
    
    def _analyze_attack_angle(self, my_hero: dict, enemy_hero: dict) -> float:
        """分析攻击角度"""
        # 简化实现
        return 0.0
    
    def _get_my_camp(self) -> str:
        """获取我方阵营 - 这里需要从上下文获取"""
        return "PLAYERCAMP_1"  # 简化实现


class AdaptiveRewardWeights:
    """自适应奖励权重管理器 - 基于论文洞察的动态权重调整"""
    
    def __init__(self):
        self.base_weights = {
            # 战略层（稀疏，高权重）
            'strategic': {
                'tower_destroy': 15.0,
                'hero_kill': 8.0,
                'hero_death': -10.0,
                'game_victory': 20.0
            },
            
            # 战术层（中密度）
            'tactical': {
                'perfect_combo_s2_s1_aa': 5.0,
                'enhanced_aa_after_s1': 2.5,
                'skill_hit_quality': 1.5,
                'kiting_execution': 2.0,
                'positioning_quality': 1.8,
                'safe_farming': 1.2
            },
            
            # 操作层（高密度，低权重）
            'operational': {
                'last_hit_accuracy': 0.4,
                'skill_accuracy': 0.3,
                'movement_efficiency': 0.2,
                'resource_management': 0.3
            }
        }
        
        self.game_phase_multipliers = {
            'early': {'strategic': 0.7, 'tactical': 1.2, 'operational': 1.1},
            'mid': {'strategic': 1.0, 'tactical': 1.0, 'operational': 1.0},
            'late': {'strategic': 1.3, 'tactical': 0.8, 'operational': 0.9}
        }
    
    def get_adaptive_weights(self, game_time_seconds: float, my_hero: dict, enemy_hero: dict) -> dict:
        """获取自适应权重"""
        # 确定游戏阶段
        game_phase = self._determine_game_phase(game_time_seconds)
        
        # 获取基础权重
        weights = {}
        for category, category_weights in self.base_weights.items():
            weights[category] = {}
            phase_multiplier = self.game_phase_multipliers[game_phase][category]
            
            for reward_type, base_weight in category_weights.items():
                # 应用阶段调整
                adjusted_weight = base_weight * phase_multiplier
                
                # 应用情境调整
                situational_multiplier = self._get_situational_multiplier(
                    reward_type, my_hero, enemy_hero
                )
                
                weights[category][reward_type] = adjusted_weight * situational_multiplier
        
        return weights
    
    def _determine_game_phase(self, game_time_seconds: float) -> str:
        """确定游戏阶段"""
        if game_time_seconds < 180:  # 前3分钟
            return 'early'
        elif game_time_seconds < 480:  # 3-8分钟
            return 'mid'
        else:  # 8分钟后
            return 'late'
    
    def _get_situational_multiplier(self, reward_type: str, my_hero: dict, enemy_hero: dict) -> float:
        """获取情境调整倍数"""
        if not my_hero or not enemy_hero:
            return 1.0
            
        # 血量情况调整
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        
        # 根据血量情况调整权重
        if reward_type in ['kiting_execution', 'positioning_quality']:
            if my_hp_ratio < 0.3:  # 低血量时更重视安全
                return 1.5
            elif enemy_hp_ratio < 0.3:  # 敌人低血量时更重视输出
                return 1.2
        
        elif reward_type == 'perfect_combo_s2_s1_aa':
            if enemy_hp_ratio < 0.5:  # 敌人血量不满时连招更有价值
                return 1.3
        
        return 1.0
    
    def _get_hp_ratio(self, hero: dict) -> float:
        """获取血量比例"""
        actor_state = hero.get("actor_state", {})
        hp = float(actor_state.get("hp", 0))
        max_hp = float(actor_state.get("max_hp", 1))
        return hp / max(max_hp, 1.0)


class EconomicGameAnalyzer:
    """经济博弈分析器 - 实现零和博弈的经济决策系统"""
    
    def __init__(self):
        self.economic_history = deque(maxlen=100)  # 经济历史记录
        self.decision_history = deque(maxlen=50)   # 决策历史记录
        
    def analyze_economic_situation(self, my_hero: dict, enemy_hero: dict, 
                                 frame_state: dict) -> Dict[str, float]:
        """分析当前经济博弈情况"""
        analysis = {}
        
        # 基础经济数据
        my_money = float(my_hero.get("money", 0))
        enemy_money = float(enemy_hero.get("money", 0))
        money_delta = my_money - enemy_money
        
        # 位置信息
        my_pos = self._get_position(my_hero)
        enemy_pos = self._get_position(enemy_hero)
        distance = math.sqrt((my_pos[0] - enemy_pos[0])**2 + (my_pos[1] - enemy_pos[1])**2)
        actor_state = my_hero.get("actor_state", {})
        if isinstance(actor_state, dict):
            attack_range = float(actor_state.get("attack_range", 600))
        else:
            attack_range = 600.0
        
        # 核心决策分析
        analysis['money_delta'] = money_delta
        analysis['economic_advantage'] = money_delta > 0
        analysis['in_attack_range'] = distance <= attack_range
        analysis['should_attack_hero'] = money_delta > 0 and distance <= attack_range
        analysis['should_farm_minions'] = money_delta > 0 and distance > attack_range
        analysis['need_defensive_farming'] = money_delta < -500
        
        # 风险评估
        analysis['economic_risk_level'] = self._assess_economic_risk(
            my_hero, enemy_hero, money_delta, distance
        )
        
        # 机会评估
        analysis['farming_opportunity'] = self._assess_farming_opportunity(
            my_hero, frame_state
        )
        
        # 压制机会
        analysis['suppress_opportunity'] = self._assess_suppress_opportunity(
            money_delta, distance, attack_range
        )
        
        # 记录历史
        self.economic_history.append({
            'money_delta': money_delta,
            'distance': distance,
            'decision': self._get_recommended_decision(analysis)
        })
        
        return analysis
    
    def get_economic_strategy_weights(self, analysis: Dict[str, float]) -> Dict[str, float]:
        """根据经济博弈分析获取策略权重"""
        weights = {
            'hero_attack_priority': 0.0,
            'minion_farming_priority': 0.0,
            'defensive_farming_priority': 0.0,
            'equipment_priority': 0.0,
            'risk_management_priority': 0.0
        }
        
        money_delta = analysis.get('money_delta', 0)
        
        if analysis.get('should_attack_hero', False):
            # 经济优势且敌人在攻击范围内 - 优先攻击英雄
            weights['hero_attack_priority'] = min(2.0 + money_delta / 1000.0, 4.0)
            weights['minion_farming_priority'] = 0.3  # 降低小兵优先级
            
        elif analysis.get('should_farm_minions', False):
            # 经济优势但敌人不在攻击范围内 - 优先发育
            weights['minion_farming_priority'] = min(1.5 + money_delta / 2000.0, 3.0)
            weights['hero_attack_priority'] = 0.5  # 保持一定攻击性
            
        elif analysis.get('need_defensive_farming', False):
            # 经济劣势 - 保守发育
            weights['defensive_farming_priority'] = min(2.0 + abs(money_delta) / 1500.0, 3.5)
            weights['risk_management_priority'] = 2.0
            weights['hero_attack_priority'] = 0.2  # 大幅降低攻击性
        
        # 装备购买权重
        current_money = analysis.get('current_money', 0)
        if current_money > 1500:
            weights['equipment_priority'] = 1.0 + current_money / 3000.0
        
        return weights
    
    def get_advanced_economic_rewards(self) -> Dict[str, Dict[str, float]]:
        """获取高级经济奖励配置"""
        return {
            # 零和博弈核心奖励
            'zero_sum_game': {
                'economic_hero_priority': 2.5,      # 经济优势时攻击英雄
                'economic_farming_priority': 1.8,   # 经济优势时发育
                'economic_catchup_farming': 2.2,    # 经济劣势时追赶发育
                'economic_safe_farming': 1.5,       # 安全发育
            },
            
            # 经济效率奖励
            'economic_efficiency': {
                'last_hit_accuracy': 1.0,           # 补刀准确性
                'farming_speed': 0.8,               # 发育速度
                'economic_growth_rate': 1.2,        # 经济增长率
                'equipment_timing': 0.6,            # 装备购买时机
            },
            
            # 经济战术奖励
            'economic_tactics': {
                'economic_pressure': 1.5,           # 经济压制
                'economic_defense': 1.3,            # 经济防守
                'resource_allocation': 0.9,         # 资源分配
                'economic_comeback': 2.0,           # 经济翻盘
            },
            
            # 惩罚机制
            'economic_penalties': {
                'missed_farming_opportunity': -0.8, # 错失发育机会
                'inefficient_resource_use': -0.5,   # 资源使用效率低
                'poor_economic_decision': -1.0,     # 错误经济决策
                'economic_waste': -0.3,             # 经济浪费
            }
        }
    
    def _get_position(self, hero: dict) -> tuple:
        """获取英雄位置"""
        pos = hero.get("actor_state", {}).get("location", {"x": 0, "z": 0})
        return (float(pos.get("x", 0)), float(pos.get("z", 0)))
    
    def _assess_economic_risk(self, my_hero: dict, enemy_hero: dict, 
                            money_delta: float, distance: float) -> float:
        """评估经济风险等级"""
        risk_factors = []
        
        # 经济劣势风险
        if money_delta < -1000:
            risk_factors.append(0.4)
        elif money_delta < -500:
            risk_factors.append(0.2)
        
        # 血量风险
        my_hp_ratio = self._get_hp_ratio(my_hero)
        enemy_hp_ratio = self._get_hp_ratio(enemy_hero)
        if my_hp_ratio < enemy_hp_ratio - 0.2:
            risk_factors.append(0.3)
        
        # 距离风险
        if distance < 400:  # 过于接近
            risk_factors.append(0.2)
        
        return sum(risk_factors)
    
    def _assess_farming_opportunity(self, my_hero: dict, frame_state: dict) -> float:
        """评估发育机会"""
        my_pos = self._get_position(my_hero)
        npcs = frame_state.get('npc_states', [])
        
        farming_score = 0.0
        minion_count = 0
        
        for npc in npcs:
            if 'SOLDIER' in npc.get('sub_type', ''):
                npc_pos = npc.get('location', {})
                if npc_pos:
                    npc_position = (float(npc_pos.get('x', 0)), float(npc_pos.get('z', 0)))
                    dist = math.sqrt((my_pos[0] - npc_position[0])**2 + (my_pos[1] - npc_position[1])**2)
                    
                    if dist <= 1200:  # 在发育范围内
                        minion_count += 1
                        
                        # 低血量小兵更有价值
                        hp_ratio = float(npc.get('hp', 0)) / max(float(npc.get('max_hp', 1)), 1)
                        if hp_ratio < 0.3:
                            farming_score += 0.3
                        elif hp_ratio < 0.6:
                            farming_score += 0.2
                        else:
                            farming_score += 0.1
        
        return min(farming_score, 1.0)
    
    def _assess_suppress_opportunity(self, money_delta: float, distance: float, 
                                   attack_range: float) -> float:
        """评估压制机会"""
        if money_delta <= 1000:  # 经济优势不够大
            return 0.0
        
        # 经济优势越大，压制机会越高
        economic_advantage = min(money_delta / 2000.0, 1.0)
        
        # 距离因素
        if distance <= attack_range:
            distance_factor = 1.0
        elif distance <= attack_range * 1.5:
            distance_factor = 0.7
        else:
            distance_factor = 0.3
        
        return economic_advantage * distance_factor
    
    def _get_recommended_decision(self, analysis: Dict[str, float]) -> str:
        """获取推荐决策"""
        if analysis.get('should_attack_hero', False):
            return 'attack_hero'
        elif analysis.get('should_farm_minions', False):
            return 'farm_minions'
        elif analysis.get('need_defensive_farming', False):
            return 'defensive_farming'
        else:
            return 'balanced_play'
    
    def _get_hp_ratio(self, hero: dict) -> float:
        """获取血量比例"""
        actor_state = hero.get("actor_state", {})
        hp = float(actor_state.get("hp", 0))
        max_hp = float(actor_state.get("max_hp", 1))
        return hp / max(max_hp, 1.0)
