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

        self.share_obs = np.zeros((self.episode_length + 1, share_obs_space), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, obs_space), dtype=np.float32)

        self.value_preds = np.zeros((self.episode_length + 1, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, 1), dtype=np.float32)

        self.actions = np.zeros((self.episode_length, act_space), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, act_space), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, 1), dtype=np.float32)

        self.step = 0

    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, rewards):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()

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
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1])  - value_normalizer.denormalize(
                            self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda  * gae
                    self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1]  - \
                                self.value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda  * gae
                    self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = self.returns[step + 1] * self.gamma  + self.rewards[step]

    def sample(self):
        done = np.zeros_like(self.rewards)

        return (self.obs, self.share_obs, self.actions, self.action_log_probs, self.rewards, self.value_preds, self.returns, done)

    def feed_forward_generator(self, advantages, mini_batch_size):
        batch_size = self.episode_length
        num_mini_batch = batch_size // mini_batch_size

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1]
        obs = self.obs[:-1]
        actions = self.actions
        value_preds = self.value_preds[:-1]
        returns = self.returns[:-1]
        action_log_probs = self.action_log_probs
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch,  actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ




