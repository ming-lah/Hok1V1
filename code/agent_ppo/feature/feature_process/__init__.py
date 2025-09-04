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
from agent_ppo.feature.feature_process.sunshangxiang_enhanced_processor import SunShangxiangEnhancedProcessor
from agent_ppo.feature.feature_process.comprehensive_feature_processor import ComprehensiveFeatureProcessor
from agent_ppo.feature.tower_strategic_processor import TowerStrategicProcessor
from agent_ppo.feature.minion_strategic_processor import MinionStrategicProcessor
from agent_ppo.feature.bush_tactical_processor import BushTacticalProcessor
import numpy as np
from collections import deque
import time


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        
        # 新增：孙尚香专用增强特征处理器
        self.sunshangxiang_processor = SunShangxiangEnhancedProcessor(camp)
        
        # 新增：全面特征处理器 (基于深度RL分析)
        self.comprehensive_processor = ComprehensiveFeatureProcessor(camp)
        
        # 新增：防御塔战略处理器 (基于防御塔核心分析)
        self.tower_strategic_processor = TowerStrategicProcessor(camp)
        
        # 新增：兵线战略处理器 (基于兵线战略分析)
        self.minion_strategic_processor = MinionStrategicProcessor(camp)
        
        # 新增：草丛战术处理器 (基于草丛战术分析)
        self.bush_tactical_processor = BushTacticalProcessor(camp)
        
        # 时序特征缓存
        self.feature_history = deque(maxlen=16)  # LSTM时间步长
        self.last_frame_no = -1
        
        # 特征处理模式选择
        self.use_comprehensive_features = True  # 默认使用全面特征处理
        self.use_tower_strategic_features = True  # 默认使用防御塔战略特征
        self.use_minion_strategic_features = True  # 默认使用兵线战略特征
        self.use_bush_tactical_features = True  # 默认使用草丛战术特征

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.sunshangxiang_processor = SunShangxiangEnhancedProcessor(camp)
        self.comprehensive_processor = ComprehensiveFeatureProcessor(camp)
        self.tower_strategic_processor = TowerStrategicProcessor(camp)
        self.minion_strategic_processor = MinionStrategicProcessor(camp)
        self.bush_tactical_processor = BushTacticalProcessor(camp)
        
        # 重置时序缓存
        self.feature_history.clear()
        self.last_frame_no = -1

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        frame_no = frame_state.get("frameNo", 0)

        if self.use_comprehensive_features:
            # 使用全面特征处理器 (基于深度RL分析的125维特征)
            comprehensive_features = self.comprehensive_processor.extract_comprehensive_features(
                observation, frame_no
            )
            
            # 追加战略特征
            final_features = comprehensive_features.tolist()  # 125维基础
            
            # 如果启用防御塔战略特征，则追加防御塔特征
            if self.use_tower_strategic_features:
                tower_features = self.tower_strategic_processor.extract_tower_strategic_features(
                    observation, frame_no
                )
                final_features.extend(tower_features)  # +35维 = 160维
            
            # 如果启用兵线战略特征，则追加兵线特征
            if self.use_minion_strategic_features:
                minion_features = self.minion_strategic_processor.extract_minion_strategic_features(
                    observation, frame_no
                )
                final_features.extend(minion_features)  # +40维 = 200维
            
            # 如果启用草丛战术特征，则追加草丛特征
            if self.use_bush_tactical_features:
                bush_features = self.bush_tactical_processor.extract_bush_tactical_features(
                    observation, frame_no
                )
                final_features.extend(bush_features)  # +25维 = 225维 (实际217维)
            
            return final_features
        else:
            # 使用原有的特征处理方式
            # 基础特征处理
            main_camp_hero_vector_feature = self.process_hero_feature(frame_state)
            organ_feature = self.process_organ_feature(frame_state)
            
            # 孙尚香专用增强特征
            sunshangxiang_enhanced_features = self.sunshangxiang_processor.extract_enhanced_features(
                observation, frame_no
            )

            # 特征融合 - 基于论文洞察的分层特征设计
            combined_features = (
                main_camp_hero_vector_feature + 
                organ_feature + 
                sunshangxiang_enhanced_features
            )
            
            # 时序特征处理 (基于论文2的LSTM时序建模)
            temporal_features = self._process_temporal_features(combined_features, frame_no)
            
            return temporal_features
        
    def _process_temporal_features(self, current_features, frame_no):
        """处理时序特征，为LSTM网络准备数据"""
        # 如果是新的一帧，添加到历史中
        if frame_no != self.last_frame_no:
            self.feature_history.append(np.array(current_features))
            self.last_frame_no = frame_no
        
        # 如果历史不足，用零填充
        if len(self.feature_history) < 16:
            padded_history = []
            # 用第一个特征填充前面的空位
            first_feature = self.feature_history[0] if self.feature_history else np.zeros_like(current_features)
            for _ in range(16 - len(self.feature_history)):
                padded_history.append(first_feature)
            padded_history.extend(list(self.feature_history))
            return np.array(padded_history)
        
        return np.array(list(self.feature_history))
