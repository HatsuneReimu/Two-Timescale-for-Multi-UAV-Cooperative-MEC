import torch
import numpy as np
from CAMRL.algorithms.utils.util import check


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer(object):
    def __init__(self, args, obs_space, share_obs_space, act_space):
        self.episode_length = args.episode_length
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm

        self.aver_episode_costs = np.zeros((self.episode_length, 1),dtype=np.float32)
        self.share_obs = np.zeros((self.episode_length + 1, share_obs_space), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, obs_space), dtype=np.float32)

        self.value_preds = np.zeros((self.episode_length + 1, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, 1), dtype=np.float32)

        self.actions = np.zeros((self.episode_length, act_space), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, act_space), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, 1), dtype=np.float32)

        self.costs = np.zeros_like(self.rewards)
        self.cost_preds = np.zeros_like(self.value_preds)
        self.cost_returns = np.zeros_like(self.returns)

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def return_aver_insert(self, aver_episode_costs):
        self.aver_episode_costs = aver_episode_costs.copy()

    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, rewards, costs, cost_preds, aver_episode_costs = 0):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.costs[self.step] = costs.copy()
        self.cost_preds[self.step] = cost_preds.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1])  - value_normalizer.denormalize(self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda  * gae
                    self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] - self.value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda  * gae
                    self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = self.returns[step + 1] * self.gamma  + self.rewards[step]

    def compute_cost_returns(self, next_cost, value_normalizer=None):
        if self._use_gae:
            self.cost_preds[-1] = next_cost
            gae = 0
            for step in reversed(range(self.costs.shape[0])):
                if self._use_popart or self._use_valuenorm:
                    delta = self.costs[step] + self.gamma * value_normalizer.denormalize(self.cost_preds[step + 1])  - value_normalizer.denormalize(self.cost_preds[step])
                    gae = delta + self.gamma * self.gae_lambda  * gae
                    self.cost_returns[step] = gae + value_normalizer.denormalize(self.cost_preds[step])
                else:
                    delta = self.costs[step] + self.gamma * self.cost_preds[step + 1] - self.cost_preds[step]
                    gae = delta + self.gamma * self.gae_lambda  * gae
                    self.cost_returns[step] = gae + self.cost_preds[step]
        else:
            self.cost_returns[-1] = next_cost
            for step in reversed(range(self.costs.shape[0])):
                self.cost_returns[step] = self.cost_returns[step + 1] * self.gamma  + self.costs[step]


    def feed_forward_generator(self, advantages, cost_adv, mini_batch_size):
        batch_size = self.episode_length
        num_mini_batch = batch_size // mini_batch_size

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1]
        obs = self.obs[:-1]
        actions = self.actions
        value_preds = self.value_preds[:-1]
        returns = self.returns[:-1]
        cost_preds = self.cost_preds[:-1]
        cost_returns = self.cost_returns[:-1]
        action_log_probs = self.action_log_probs
        aver_episode_costs = self.aver_episode_costs
        if self.factor is not None:
            factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)
        cost_adv = cost_adv.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            cost_preds_batch = cost_preds[indices]
            cost_return_batch = cost_returns[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices]
            cost_adv_targ = cost_adv[indices]

            factor_batch = factor[indices]
            yield share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, factor_batch, cost_preds_batch, cost_return_batch, cost_adv_targ, aver_episode_costs




