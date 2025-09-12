#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.wave_process import WaveProcess


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.wave_process = WaveProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.wave_process = WaveProcess(camp)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_wave_feature(self, frame_state, hero_info):
        return self.wave_process.process_wave_features(frame_state, hero_info)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        # 获取英雄特征
        main_camp_hero_vector_feature = self.process_hero_feature(frame_state)
        
        # 获取器官/防御塔特征
        organ_feature = self.process_organ_feature(frame_state)
        
        # 获取当前己方英雄信息用于兵线特征计算
        hero_info = None
        for hero in frame_state.get("hero_states", []):
            if hero.get("actor_state", {}).get("camp") == self.camp:
                hero_info = hero
                break
        
        # 获取兵线宏观特征
        wave_feature = self.process_wave_feature(frame_state, hero_info)

        # 拼接所有特征
        feature = main_camp_hero_vector_feature + organ_feature + wave_feature

        return feature
