#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoK 1v1 Reward (8 + forward_progress) aligned to the doc:

dense : hp_point, tower_hp_point, money, ep_rate, exp, forward_progress
sparse: death, kill, last_hit

- frame_action: 支持“一维向量”，从 GameConfig.FRAME_ACTION_VECTOR_MAP 按索引读取；
  可配置累计/逐帧、二值/计数。
- 密集项口径：
  * SELF_DENSE = {hp_point, ep_rate, money, exp, forward_progress}
      -> 仅取“我方本帧增量 my_cur - my_last”
  * ADV_DENSE  = {tower_hp_point}
      -> 取对抗优势的增量 (my - en)_t - (my - en)_{t-1}
- 稀疏项口径：
  * SPARSE = {death, kill, last_hit} -> “我方本帧事件次数” = my_cur - my_last
- forward_progress:
  * 以“我方塔→敌方塔”的单位向量为前进轴 u，英雄位置 p 的投影 s = <p, u>
  * 本帧奖励 = s_t - s_{t-1}，可配置只计正向、步长上限
- 仍保留 TIME_SCALE_ARG 衰减。
"""

from typing import Dict, Any, Tuple, List
import math

from agent_ppo.conf.conf import GameConfig


# --------------------------- data struct --------------------------- #
class RewardStruct:
    def __init__(self, m_weight: float = 0.0):
        self.cur_frame_value: float = 0.0
        self.last_frame_value: float = 0.0
        self.value: float = 0.0
        self.weight: float = m_weight


def init_calc_frame_map() -> Dict[str, RewardStruct]:
    m: Dict[str, RewardStruct] = {}
    for key, w in GameConfig.REWARD_WEIGHT_DICT.items():
        m[key] = RewardStruct(float(w))
    return m


# ----------------------------- manager ----------------------------- #
class GameRewardManager:
    """
    使用方法：
      mgr = GameRewardManager(main_player_id=<你的我方 player_id 或 runtime_id>)
      d = mgr.result(frame_state)  # 返回各子项和 reward_sum
    """

    SELF_DENSE = {"hp_point", "ep_rate", "money", "exp", "forward_progress"}
    ADV_DENSE  = {"tower_hp_point"}
    SPARSE     = {"death", "kill", "last_hit"}

    def __init__(self, main_player_id: int):
        # 允许用 player_id 或 runtime_id 识别自己
        self.main_player_id = int(main_player_id)
        self.main_hero_camp = None  # 在首帧识别
        # 三套 map：我方、敌方、（只是用作权重容器与输出承载）
        self.m_main_calc_frame_map  = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_cur_calc_frame_map   = init_calc_frame_map()

        self.time_scale_arg = getattr(GameConfig, "TIME_SCALE_ARG", 0)
        # 事件向量累计缓存（用于做差分）
        self._last_event_cum = {"my_kill_hero": 0, "my_death": 0, "my_last_hit_soldier": 0}
        self._printed_probe = False

    # ======================= public interface ======================= #
    def result(self, frame_state: Dict[str, Any]) -> Dict[str, float]:
        self._frame_data_process(frame_state)
        out: Dict[str, float] = {}
        self._combine_reward(frame_state, out)

        # 时间衰减（可选）
        frame_no = int(frame_state.get("frameNo", 0))
        if self.time_scale_arg and self.time_scale_arg > 0:
            decay = math.pow(0.6, 1.0 * frame_no / float(self.time_scale_arg))
            for k in out:
                out[k] *= decay
        return out

    # ============================ helpers ============================ #
    @staticmethod
    def _hp_ratio(state: Dict[str, Any]) -> float:
        if not isinstance(state, dict):
            return 0.0
        hp = float(state.get("hp", 0.0))
        mx = float(state.get("max_hp", 0.0))
        return hp / mx if mx > 0 else 0.0

    @staticmethod
    def _actor_state_from_hero(hero: Dict[str, Any]) -> Dict[str, Any]:
        return (hero or {}).get("actor_state") or {}

    @staticmethod
    def _camp_of_hero(hero: Dict[str, Any]):
        return ((hero or {}).get("actor_state") or {}).get("camp", None)

    @staticmethod
    def _get_gold(astate: Dict[str, Any]) -> float:
        return float(astate.get("gold", astate.get("money", 0.0)) or 0.0)

    @staticmethod
    def _get_ep_rate(astate: Dict[str, Any]) -> float:
        v = astate.get("values") or {}
        ep, mx = float(v.get("ep", 0.0)), float(v.get("max_ep", 1.0))
        return 0.0 if mx <= 0 else ep / mx

    @staticmethod
    def _get_exp(hero: Dict[str, Any]) -> float:
        try:
            return float((hero or {}).get("exp", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _is_tower(obj: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        for k in ("sub_type", "actor_sub_type", "actor_type", "type"):
            v = obj.get(k)
            if isinstance(v, str) and v in ("ACTOR_SUB_TOWER", "TOWER"):
                return True
        return False

    @staticmethod
    def _xy_from(obj: Dict[str, Any]) -> Tuple[float, float]:
        """从 hero.actor_state.location 或 npc.location 取 (x,z)"""
        if not isinstance(obj, dict):
            return 0.0, 0.0
        loc = (obj.get("actor_state") or {}).get("location") if "actor_state" in obj else obj.get("location")
        if not isinstance(loc, dict):
            return 0.0, 0.0
        return float(loc.get("x", 0.0)), float(loc.get("z", 0.0))

    def _identify_camps(self, frame_state: Dict[str, Any]):
        """在首帧根据 player_id 或 runtime_id 识别我方阵营。"""
        if self.main_hero_camp is not None:
            return
        for h in frame_state.get("hero_states", []) or []:
            pid = h.get("player_id", None)
            rt  = ((h.get("actor_state") or {}).get("runtime_id", None))
            if pid == self.main_player_id or rt == self.main_player_id:
                self.main_hero_camp = self._camp_of_hero(h)
                break
        # 兜底
        if self.main_hero_camp is None:
            self.main_hero_camp = frame_state.get("player_camp", None)

    def _split_heroes(self, frame_state: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """返回 (my_hero, enemy_hero)"""
        me, en = None, None
        for h in frame_state.get("hero_states", []) or []:
            c = self._camp_of_hero(h)
            if c == self.main_hero_camp:
                me = h
            else:
                en = h
        return me, en

    def _collect_towers(self, frame_state: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """收集两边全部塔对象列表 (my_towers, enemy_towers)"""
        my_list, en_list = [], []
        for u in frame_state.get("npc_states", []) or []:
            if not self._is_tower(u):
                continue
            if u.get("camp") == self.main_hero_camp:
                my_list.append(u)
            else:
                en_list.append(u)
        return my_list, en_list

    def _lane_axis(self, frame_state: Dict[str, Any]) -> Tuple[float, float]:
        """
        用“我方塔群质心 -> 敌方塔群质心”的方向为 lane axis。
        失败时回退为 (0,1)（朝 +z 方向）。
        """
        my_ts, en_ts = self._collect_towers(frame_state)
        def centroid(lst: List[Dict]) -> Tuple[float, float]:
            if not lst:
                return 0.0, 0.0
            sx = sz = 0.0
            for t in lst:
                x, z = self._xy_from(t)
                sx += x; sz += z
            n = max(1, len(lst))
            return sx / n, sz / n

        x1, z1 = centroid(my_ts)
        x2, z2 = centroid(en_ts)
        dx, dz = (x2 - x1), (z2 - z1)
        n = math.hypot(dx, dz)
        if n <= 1e-6:
            return 0.0, 1.0
        return dx / n, dz / n

    # ======================== frame processing ======================== #
    def _frame_data_process_one_side(self, calc_map: Dict[str, RewardStruct],
                                     frame_state: Dict[str, Any], for_my_side: bool):
        """
        写入当前帧的“我方口径”数值到 calc_map 的 cur_frame_value。
        稀疏事件：由 _read_events_from_vector() 负责。
        """
        # 标定阵营
        self._identify_camps(frame_state)

        my_hero, _ = self._split_heroes(frame_state)
        astate = self._actor_state_from_hero(my_hero)

        # --------- dense: 当前绝对值 --------- #
        cur_vals = {
            "hp_point":       self._hp_ratio(astate),
            "tower_hp_point": 0.0,  # 稍后再填（需要从 npc 直接取）
            "money":          self._get_gold(astate),
            "ep_rate":        self._get_ep_rate(astate),
            "exp":            self._get_exp(my_hero),
            "forward_progress": 0.0,  # 投影值，稍后填
        }

        # 我方塔血比例
        #（直接从 npc_states 取塔对象）
        my_towers, _ = self._collect_towers(frame_state)
        # 没塔时置 0，有多座时取平均
        if my_towers:
            ratios = []
            for t in my_towers:
                ratios.append(self._hp_ratio(t))
            cur_vals["tower_hp_point"] = sum(ratios) / max(1, len(ratios))

        # forward_progress: 位置在 lane axis 上的投影值
        if "forward_progress" in calc_map:
            ux, uz = self._lane_axis(frame_state)
            hx, hz = self._xy_from(my_hero)
            cur_vals["forward_progress"] = hx * ux + hz * uz  # 绝对投影；奖励阶段做“本帧增量”

        # --------- sparse: 本帧事件（由向量适配） --------- #
        mk, md, mlh = self._read_events_from_vector(frame_state)
        cur_vals.update({
            "kill":     float(mk),
            "death":    float(md),
            "last_hit": float(mlh),
        })

        # 写入：last <- cur, cur <- new
        for name, rs in calc_map.items():
            rs.last_frame_value = rs.cur_frame_value
            rs.cur_frame_value  = float(cur_vals.get(name, 0.0))

    def _frame_data_process(self, frame_state: Dict[str, Any]):
        """分别按我方/敌方口径写两套 map。"""
        # 我方
        self._frame_data_process_one_side(self.m_main_calc_frame_map, frame_state, for_my_side=True)

        # 敌方：临时翻转视角（只在读取时翻，不改 self.main_hero_camp）
        orig_camp = self.main_hero_camp
        enemy_camp = None
        for h in frame_state.get("hero_states", []) or []:
            c = self._camp_of_hero(h)
            if c is not None and c != orig_camp:
                enemy_camp = c
                break
        self.main_hero_camp = enemy_camp if enemy_camp is not None else -9999
        self._frame_data_process_one_side(self.m_enemy_calc_frame_map, frame_state, for_my_side=False)
        self.main_hero_camp = orig_camp

    # ========================= event (vector) ========================= #
    def _read_events_from_vector(self, frame_state: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        从一维向量 frame_action 中读出 (kill, death, last_hit)。
        需要在 GameConfig 中配置：
          FRAME_ACTION_VECTOR_MAP = {"my_kill_hero":k, "my_death":k, "my_last_hit_soldier":k}
          FRAME_ACTION_VECTOR_IS_CUMULATIVE = True/False
          FRAME_ACTION_VECTOR_IS_BINARY     = True/False
        """
        fa = frame_state.get("frame_action", None)

        # 首帧打印一次探针，确认形状
        if not self._printed_probe:
            try:
                import numpy as np
                shape = getattr(fa, "shape", None)
                print(f"[reward] frame_action probe: type={type(fa)}, shape={shape}, len={len(fa) if isinstance(fa,(list,tuple)) else 'n/a'}")
            except Exception:
                print(f"[reward] frame_action probe: type={type(fa)}")
            self._printed_probe = True

        if fa is None:
            return 0, 0, 0

        try:
            import numpy as np
            if isinstance(fa, np.ndarray):
                if fa.ndim == 0:
                    return 0, 0, 0
                if fa.ndim > 1:
                    fa = fa.reshape(-1)
            elif not isinstance(fa, (list, tuple)):
                return 0, 0, 0
        except Exception:
            if not isinstance(fa, (list, tuple)):
                return 0, 0, 0

        vec = list(fa)
        n = len(vec)

        mp = getattr(GameConfig, "FRAME_ACTION_VECTOR_MAP", {})
        if not mp:
            return 0, 0, 0

        def geti(key: str) -> int:
            idx = mp.get(key, -1)
            if 0 <= idx < n:
                v = float(vec[idx])
                if getattr(GameConfig, "FRAME_ACTION_VECTOR_IS_BINARY", False):
                    return int(v > 0.5)
                return int(round(v))
            return 0

        kill_c  = geti("my_kill_hero")
        death_c = geti("my_death")
        lh_c    = geti("my_last_hit_soldier")

        if getattr(GameConfig, "FRAME_ACTION_VECTOR_IS_CUMULATIVE", True):
            last = self._last_event_cum
            d_k  = max(0, kill_c  - last.get("my_kill_hero", 0))
            d_d  = max(0, death_c - last.get("my_death", 0))
            d_lh = max(0, lh_c    - last.get("my_last_hit_soldier", 0))
            self._last_event_cum = {
                "my_kill_hero": kill_c,
                "my_death": death_c,
                "my_last_hit_soldier": lh_c,
            }
            return d_k, d_d, d_lh

        return max(0, kill_c), max(0, death_c), max(0, lh_c)

    # ======================== reward combining ======================== #
    def _combine_reward(self, frame_state: Dict[str, Any], out: Dict[str, float]):
        out.clear()
        reward_sum = 0.0

        # 读 forward_progress 的裁剪/上限配置
        clip_neg = bool(getattr(GameConfig, "FORWARD_PROGRESS_CLIP_NEGATIVE", True))
        step_cap = float(getattr(GameConfig, "FORWARD_PROGRESS_STEP_CAP", 0.0))  # 0 表示不限制

        for name, rs_out in self.m_cur_calc_frame_map.items():
            w = rs_out.weight or 0.0

            if name in self.ADV_DENSE:
                my_cur  = self.m_main_calc_frame_map[name].cur_frame_value
                en_cur  = self.m_enemy_calc_frame_map[name].cur_frame_value
                my_last = self.m_main_calc_frame_map[name].last_frame_value
                en_last = self.m_enemy_calc_frame_map[name].last_frame_value
                rs_out.value = (my_cur - en_cur) - (my_last - en_last)

            elif name in self.SELF_DENSE:
                my_cur  = self.m_main_calc_frame_map[name].cur_frame_value
                my_last = self.m_main_calc_frame_map[name].last_frame_value
                rs_out.value = my_cur - my_last

                # 针对 forward_progress 的附加约束（只奖励前进、步长上限）
                if name == "forward_progress":
                    if clip_neg and rs_out.value < 0.0:
                        rs_out.value = 0.0
                    if step_cap and step_cap > 0.0:
                        if rs_out.value > step_cap:
                            rs_out.value = step_cap
                        elif rs_out.value < -step_cap:
                            rs_out.value = -step_cap

            elif name in self.SPARSE:
                my_cur  = self.m_main_calc_frame_map[name].cur_frame_value
                my_last = self.m_main_calc_frame_map[name].last_frame_value
                rs_out.value = my_cur - my_last  # 本帧事件数（0/1）

            else:
                rs_out.value = 0.0

            out[name] = rs_out.value
            reward_sum += rs_out.value * w

        out["reward_sum"] = reward_sum
