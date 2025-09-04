#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config
from typing import Dict, List, Tuple, Optional


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.last_report_monitor_time = 0
        
        # 双剪切PPO参数 - 基于论文2的大规模训练优化
        self.use_dual_clip_ppo = getattr(Config, 'USE_DUAL_CLIP_PPO', True)
        self.dual_clip_c = getattr(Config, 'DUAL_CLIP_C', 2.0)
        self.clip_param = getattr(Config, 'CLIP_PARAM', 0.2)
        
        # 孙尚香专用奖励权重管理器
        self.adaptive_reward_manager = AdaptiveRewardManager()
        
        # 训练统计
        self.training_stats = {
            'combo_rewards': [],
            'kiting_rewards': [],
            'positioning_rewards': [],
            'policy_losses': [],
            'value_losses': []
        }

    def learn(self, list_sample_data):
        _input_datas = torch.stack([t.npdata for t in list_sample_data], dim=0)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data = data.reshape(-1)
            data_list[i] = data.float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad clip
        # 梯度剪裁
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        # update the learning rate
        # 更新学习率
        self.scheduler.step(self.train_step)

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now
            
    def compute_dual_clip_ppo_loss(self, log_probs_old: torch.Tensor, log_probs_new: torch.Tensor, 
                                  advantages: torch.Tensor) -> torch.Tensor:
        """
        计算双剪切PPO损失 - 基于论文2的大规模训练稳定性优化
        
        Args:
            log_probs_old: 旧策略的对数概率
            log_probs_new: 新策略的对数概率  
            advantages: 优势函数值
        
        Returns:
            PPO损失值
        """
        # 计算概率比率
        ratio = torch.exp(log_probs_new - log_probs_old)
        
        # 标准PPO剪切
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        if self.use_dual_clip_ppo:
            # 双剪切：额外的下界约束，防止策略更新过于激进
            # 当优势为正且比率过大时，进一步限制更新
            surr3 = torch.max(
                surr2,
                self.dual_clip_c * advantages
            )
            # 选择最保守的更新
            policy_loss = -torch.mean(torch.min(surr1, surr3))
        else:
            # 标准PPO
            policy_loss = -torch.mean(torch.min(surr1, surr2))
        
        return policy_loss
    
    def compute_enhanced_value_loss(self, values_pred: torch.Tensor, values_target: torch.Tensor,
                                   values_old: torch.Tensor) -> torch.Tensor:
        """
        计算增强价值损失 - 结合剪切和平滑损失
        
        Args:
            values_pred: 预测价值
            values_target: 目标价值
            values_old: 旧价值预测
        
        Returns:
            价值损失
        """
        # 剪切价值损失（类似PPO策略剪切）
        value_pred_clipped = values_old + torch.clamp(
            values_pred - values_old, -self.clip_param, self.clip_param
        )
        
        # 计算两种损失
        value_loss_unclipped = F.mse_loss(values_pred, values_target)
        value_loss_clipped = F.mse_loss(value_pred_clipped, values_target)
        
        # 取最大值确保保守更新
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        
        return value_loss


class AdaptiveRewardManager:
    """自适应奖励管理器 - 基于论文洞察的动态奖励调整"""
    
    def __init__(self):
        # 奖励历史统计
        self.reward_history = {
            'combo_success_rate': [],
            'kiting_quality_avg': [],
            'positioning_score_avg': [],
            'win_rate': []
        }
        
        # 自适应权重
        self.adaptive_weights = {
            'combo_reward_weight': 1.0,
            'kiting_reward_weight': 1.0,
            'positioning_reward_weight': 1.0,
            'safety_reward_weight': 1.0
        }
        
        # 更新频率
        self.update_frequency = 100  # 每100步更新一次权重
        self.step_count = 0
        
    def update_reward_weights(self, training_metrics: Dict[str, float]):
        """根据训练指标动态调整奖励权重"""
        self.step_count += 1
        
        if self.step_count % self.update_frequency != 0:
            return
        
        # 记录历史
        for key, value in training_metrics.items():
            if key in self.reward_history:
                self.reward_history[key].append(value)
                # 保持历史长度
                if len(self.reward_history[key]) > 1000:
                    self.reward_history[key] = self.reward_history[key][-1000:]
        
        # 动态调整权重
        self._adjust_combo_weight(training_metrics)
        self._adjust_tactical_weights(training_metrics)
        
    def _adjust_combo_weight(self, metrics: Dict[str, float]):
        """调整连招奖励权重"""
        combo_success_rate = metrics.get('combo_success_rate', 0.0)
        
        if combo_success_rate < 0.3:
            # 连招成功率低，增加连招奖励权重
            self.adaptive_weights['combo_reward_weight'] *= 1.1
        elif combo_success_rate > 0.8:
            # 连招成功率高，可以降低权重，关注其他方面
            self.adaptive_weights['combo_reward_weight'] *= 0.95
        
        # 限制权重范围
        self.adaptive_weights['combo_reward_weight'] = np.clip(
            self.adaptive_weights['combo_reward_weight'], 0.5, 3.0
        )
    
    def _adjust_tactical_weights(self, metrics: Dict[str, float]):
        """调整战术奖励权重"""
        win_rate = metrics.get('win_rate', 0.5)
        kiting_quality = metrics.get('kiting_quality_avg', 0.0)
        
        if win_rate < 0.4:
            # 胜率低，增加安全性权重
            self.adaptive_weights['safety_reward_weight'] *= 1.05
            self.adaptive_weights['positioning_reward_weight'] *= 1.05
        elif win_rate > 0.6:
            # 胜率高，可以更激进
            self.adaptive_weights['kiting_reward_weight'] *= 1.05
        
        # 限制权重范围
        for key in ['kiting_reward_weight', 'positioning_reward_weight', 'safety_reward_weight']:
            self.adaptive_weights[key] = np.clip(self.adaptive_weights[key], 0.5, 2.0)
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前奖励权重"""
        return self.adaptive_weights.copy()


class SunShangxiangSpecificOptimizer:
    """孙尚香专用训练优化器 - 基于论文洞察的专用优化策略"""
    
    def __init__(self):
        # 技能使用频率统计
        self.skill_usage_stats = {
            'skill1_usage': 0,
            'skill2_usage': 0,
            'skill3_usage': 0,
            'combo_2_1_aa_usage': 0,
            'enhanced_aa_usage': 0
        }
        
        # 性能指标
        self.performance_metrics = {
            'average_game_length': 0.0,
            'damage_per_minute': 0.0,
            'kda_ratio': 1.0,
            'tower_damage_ratio': 0.0
        }
        
    def analyze_gameplay_pattern(self, game_data: Dict) -> Dict[str, float]:
        """分析游戏模式，提供训练建议"""
        analysis = {}
        
        # 技能使用分析
        total_skills = sum(self.skill_usage_stats.values())
        if total_skills > 0:
            combo_ratio = self.skill_usage_stats['combo_2_1_aa_usage'] / total_skills
            analysis['combo_execution_rate'] = combo_ratio
            
            if combo_ratio < 0.1:
                analysis['training_focus'] = 'combo_training'
            elif combo_ratio > 0.3:
                analysis['training_focus'] = 'positioning_training'
            else:
                analysis['training_focus'] = 'balanced_training'
        
        # 生存能力分析
        kda = self.performance_metrics['kda_ratio']
        if kda < 1.0:
            analysis['survival_training_needed'] = True
            analysis['recommended_kiting_weight'] = 1.5
        else:
            analysis['survival_training_needed'] = False
            analysis['recommended_kiting_weight'] = 1.0
        
        return analysis
    
    def get_curriculum_learning_schedule(self, training_step: int) -> Dict[str, float]:
        """获取课程学习调度"""
        # 分阶段训练策略
        if training_step < 10000:
            # 早期：专注基础操作
            return {
                'basic_operation_weight': 2.0,
                'combo_complexity_weight': 0.5,
                'tactical_weight': 0.8
            }
        elif training_step < 50000:
            # 中期：增加连招训练
            return {
                'basic_operation_weight': 1.5,
                'combo_complexity_weight': 1.5,
                'tactical_weight': 1.2
            }
        else:
            # 后期：全面训练
            return {
                'basic_operation_weight': 1.0,
                'combo_complexity_weight': 1.8,
                'tactical_weight': 1.5
            }
