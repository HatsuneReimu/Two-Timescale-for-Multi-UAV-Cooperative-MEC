import numpy as np
import torch
import torch.nn as nn
from CAMRL.algorithms.MAPPOPolicy_Lagr import MAPPOPolicy_L
from CAMRL.algorithms.utils.util import get_gard_norm, huber_loss, mse_loss, check
from CAMRL.algorithms.utils.popart import PopArt
from CAMRL.algorithms.utils.buffer_larg import SeparatedReplayBuffer


class MAPPO:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = MAPPOPolicy_L(args, obs_space, cent_obs_space, act_space, device)
        self.buffer = SeparatedReplayBuffer(args, obs_space, cent_obs_space, act_space)
        self._damping = 0.00001

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.mini_batch_size = args.mini_batch_size
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.gamma = args.gamma

        self.episode_length = args.episode_length
        self.num_agents = args.num_uavs

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_popart = args.use_popart
        self._use_huber_loss = args.use_huber_loss
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self.data_chunk_length = getattr(args, "data_chunk_length", None)

        self.lagrangian_coef = args.lagrangian_coef_rate # lagrangian_coef
        self.lamda_lagr = args.lamda_lagr # 0.78
        self.safety_bound = args.safety_bound # 0.2 Ant

        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None



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

    def _get_flat_grad(self, y: torch.Tensor, model: nn.Module, **kwargs) -> torch.Tensor:
        grads = torch.autograd.grad(y, model.parameters(), **kwargs, allow_unused=True)  # type: ignore
        _grads = []
        for val in grads:
            if val != None:
                _grads.append(val);

        return torch.cat([grad.reshape(-1) for grad in _grads])

    def _conjugate_gradients(self, b: torch.Tensor, flat_kl_grad: torch.Tensor, nsteps: int = 10,
                             residual_tol: float = 1e-10) -> torch.Tensor:
        x = torch.zeros_like(b)
        r, p = b.clone(), b.clone()
        rdotr = r.dot(r)
        for i in range(nsteps):
            z = self.cal_second_hessian(p, flat_kl_grad)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def cal_second_hessian(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        kl_v = (flat_kl_grad * v).sum()
        flat_kl_grad_grad = self._get_flat_grad(
            kl_v, self.policy.actor, retain_graph=True).detach()
        return flat_kl_grad_grad + v * self._damping

    def _set_from_flat_params(self, model: nn.Module, flat_params: torch.Tensor) -> nn.Module:
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        return model

    def take_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action = self.policy.act(obs_tensor, deterministic = False)
        return np.array(action)

    def ppo_update(self, sample, update_actor=True):

        share_obs_batch, obs_batch, actions_batch,value_preds_batch, return_batch, old_action_log_probs_batch, \
            adv_targ, factor_batch, cost_preds_batch, cost_returns_barch, cost_adv_targ, aver_episode_costs = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        cost_returns_barch = check(cost_returns_barch).to(**self.tpdv)
        cost_preds_batch = check(cost_preds_batch).to(**self.tpdv)

        values, cost_values, action_log_probs, dist_entropy, = self.policy.evaluate_actions(share_obs_batch,
                                                                                           obs_batch,
                                                                                           actions_batch)
        # todo: lagrangian coef
        adv_targ_hybrid =  adv_targ + self.lamda_lagr*cost_adv_targ
        # todo: lagrangian actor update step
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ_hybrid
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_hybrid

        policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # todo: update lamda_lagr
        delta_lamda_lagr = -(( abs(aver_episode_costs.mean()) - self.safety_bound) * (1 - self.gamma) )
        delta_lamda_lagr = torch.tensor(delta_lamda_lagr, dtype=torch.float32)
        R_Relu = torch.nn.ReLU()
        new_lamda_lagr = R_Relu(self.lamda_lagr - (delta_lamda_lagr * self.lagrangian_coef))
        self.lamda_lagr = new_lamda_lagr

        # todo: reward critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        # todo: cost critic update
        cost_loss = self.cal_value_loss(cost_values, cost_preds_batch, cost_returns_barch)
        self.policy.cost_optimizer.zero_grad()
        (cost_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            cost_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.max_grad_norm)
        else:
            cost_grad_norm = get_gard_norm(self.policy.cost_critic.parameters())
        self.policy.cost_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost_loss, cost_grad_norm

    def train(self, update_actor=True):
        share_obs_last = self.buffer.share_obs[-1]
        next_value = self.policy.get_values(share_obs_last)
        next_value_n = next_value.detach().cpu().numpy()
        next_cost = self.policy.get_cost_values(share_obs_last)
        next_cost_n = next_cost.detach().cpu().numpy()
        self.buffer.compute_returns(next_value_n, self.value_normalizer)
        self.buffer.compute_cost_returns(next_cost_n, self.value_normalizer)

        if self._use_popart:
            advantages = self.buffer.returns[:-1] - self.value_normalizer.denormalize(self.buffer.value_preds[:-1])
        else:
            advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        mean_adv = advantages_copy.mean()
        std_adv = advantages_copy.std()
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        if self._use_popart:
            cost_adv = self.buffer.cost_returns[:-1] - self.value_normalizer.denormalize(self.buffer.cost_preds[:-1])
        else:
            cost_adv = self.buffer.cost_returns[:-1] - self.buffer.cost_preds[:-1]

        cost_adv_copy = cost_adv.copy()
        mean_cost_adv = cost_adv_copy.mean()
        std_cost_adv = cost_adv_copy.std()
        cost_adv = (cost_adv - mean_cost_adv) / (std_cost_adv + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['cost_grad_norm'] = 0
        train_info['cost_loss'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = self.buffer.feed_forward_generator(advantages, cost_adv, self.mini_batch_size)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost_loss, cost_grad_norm \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['cost_loss'] += cost_loss.item()
                train_info['cost_grad_norm'] += cost_grad_norm

        num_updates = self.ppo_epoch * (self.episode_length / self.mini_batch_size)

        for k in train_info.keys():
            train_info[k] /= num_updates

        self.buffer.after_update()

        return train_info

