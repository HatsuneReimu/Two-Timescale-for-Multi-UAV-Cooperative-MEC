"""
使用新的缓存池，并修改了MAPPO的逻辑
"""
import numpy as np
import torch
import torch.nn as nn
from CAMRL.algorithms.MAPPOPolicy import MAPPOPolicy
from CAMRL.algorithms.utils.util import get_gard_norm, huber_loss, mse_loss, check
from CAMRL.algorithms.utils.popart import PopArt
from CAMRL.algorithms.utils.buffer_pro import SeparatedReplayBuffer


class MAPPO:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device):
        self.device = device
        self.gamma = args.gamma
        self.eps = args.policy_clip
        self.gae_lambda = args.gae_lambda
        self.ppo_epoch = args.ppo_epoch
        self.episode_length = args.episode_length
        self.mini_batch_size = args.mini_batch_size
        self.num_agents = args.num_uavs
        self.value_loss_coef = args.value_loss_coef
        self._use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.entropy_coef = args.entropy_coef

        self.clip_param = args.clip_param
        self._use_popart = args.use_popart
        self._use_huber_loss = args.use_huber_loss
        self.huber_delta = args.huber_delta
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.data_chunk_length = getattr(args, "data_chunk_length", None)

        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

        self.policy = MAPPOPolicy(args, obs_space, cent_obs_space, act_space, device)
        self.buffer = SeparatedReplayBuffer(args, obs_space, cent_obs_space, act_space)

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def take_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action = self.policy.act(obs_tensor, deterministic = False)
        return np.array(action)

    def ppo_update(self, sample, update_actor=True):

        share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy= self.policy.evaluate_actions(share_obs_batch,
                                                                                           obs_batch,
                                                                                           actions_batch)

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # todo: reward critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, update_actor=True):
        share_obs_last = self.buffer.share_obs[-1]
        next_value = self.policy.get_values(share_obs_last)
        next_value_n = next_value.detach().cpu().numpy()
        self.buffer.compute_returns(next_value_n, self.value_normalizer)

        if self._use_popart:
            advantages = self.buffer.returns[:-1] - self.value_normalizer.denormalize(self.buffer.value_preds[:-1])
        else:
            advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        mean_adv = advantages_copy.mean()
        std_adv = advantages_copy.std()
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = self.buffer.feed_forward_generator(advantages, self.mini_batch_size)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * (self.episode_length / self.mini_batch_size)

        for k in train_info.keys():
            train_info[k] /= num_updates

        self.buffer.after_update()

        return train_info

