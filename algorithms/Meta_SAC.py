import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Normal

# 设置设备 (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def functional_actor_forward(params, x):
    """
    用给定的参数字典对 SACActor 做函数式前向传播。
    不依赖 torch.func / functional_call, 兼容老版本 PyTorch。
    params 的 key 形如 'fc1.weight', 'fc1.bias', 'mu_head.weight' ...
    对 params 保持可微, 从而支持 Meta-SAC 的二阶元梯度。
    """
    x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
    x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
    mu = F.linear(x, params['mu_head.weight'], params['mu_head.bias'])
    log_std = F.linear(x, params['log_std_head.weight'], params['log_std_head.bias'])
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mu, log_std

def squash_action(mu, log_std):
    """
    根据网络输出的 mu / log_std 采样动作并计算 log_prob。
    抽成独立函数是为了能在函数式前向 (functional_call) 后复用同一套采样逻辑，
    这样 Meta-SAC 的内层前瞻更新才能对策略参数保持可微。
    """
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = log_std.exp()
    normal = Normal(mu, std)

    x_t = normal.rsample()          # 重参数化采样
    y_t = torch.tanh(x_t)           # 压缩到 [-1, 1]
    action = y_t

    log_prob = normal.log_prob(x_t)
    log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)   # tanh 雅可比修正
    log_prob = log_prob.sum(1, keepdim=True)
    return action, log_prob


class SACActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        return squash_action(mu, log_std)


class SACCritic(nn.Module):
    def __init__(self, state_dim, total_action_dim):
        super(SACCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + total_action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# 经验回放池
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, state, action, reward, next_obs, next_state, done):
        self.buffer.append((obs, state, action, reward, next_obs, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, state, action, reward, next_obs, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(obs).to(device),
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(next_obs).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).to(device)
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# Meta-SAC 单智能体容器
# ==========================================
class MetaSACAgent:
    def __init__(self, obs_dim, act_dim, state_dim, total_act_dim,
                 lr=3e-4, alpha_lr=3e-4, inner_lr=3e-4):
        # 1. Actor
        self.actor = SACActor(obs_dim, act_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # 2. 双 Critic + Target
        self.critic1 = SACCritic(state_dim, total_act_dim).to(device)
        self.critic2 = SACCritic(state_dim, total_act_dim).to(device)
        self.target_critic1 = SACCritic(state_dim, total_act_dim).to(device)
        self.target_critic2 = SACCritic(state_dim, total_act_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # 3. 温度参数 Alpha —— 在 Meta-SAC 中作为元参数，通过元梯度学习
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()

        # 内层前瞻更新使用的学习率（用于对策略做一步可微更新）
        self.inner_lr = inner_lr


class MetaSAC:
    def __init__(self, env, gamma=0.95, tau=0.01, lr=3e-4,
                 alpha_lr=3e-4, inner_lr=3e-4):
        self.num_agents = env.num_uavs
        self.gamma = gamma
        self.tau = tau
        obs_dim = 3 + 2 * env.num_users
        action_dim = 3
        state_dim = 3 * env.num_uavs + 2 * env.num_users
        total_act_dim = env.num_uavs * action_dim

        self.action_dim = action_dim

        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(
                MetaSACAgent(obs_dim, action_dim, state_dim, total_act_dim,
                             lr=lr, alpha_lr=alpha_lr, inner_lr=inner_lr)
            )

    def select_action(self, obs, evaluate=False):
        actions = []
        for i, agent in enumerate(self.agents):
            o = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                if evaluate:
                    mu, _ = agent.actor(o)
                    a = torch.tanh(mu).cpu().numpy().flatten()
                else:
                    a, _ = agent.actor.sample(o)
                    a = a.cpu().numpy().flatten()
            actions.append(a)
        return np.array(actions)

    def _joint_action_for_agent(self, i, obs_batch, use_params=None, meta_obs=False):
        """
        构造集中式 Critic 需要的联合动作:
        - 只有智能体 i 的动作带梯度 (可选地用给定的 params 做函数式前向)
        - 其它智能体动作 detach
        返回: 联合动作张量, 智能体 i 的 log_prob
        """
        curr_actions = []
        log_pi_i = None
        for j, ag in enumerate(self.agents):
            if j == i:
                if use_params is None:
                    a_j, log_pi_i = ag.actor.sample(obs_batch[:, j, :])
                else:
                    mu, log_std = functional_actor_forward(use_params, obs_batch[:, j, :])
                    a_j, log_pi_i = squash_action(mu, log_std)
            else:
                with torch.no_grad():
                    a_j, _ = ag.actor.sample(obs_batch[:, j, :])
            curr_actions.append(a_j)
        return torch.cat(curr_actions, dim=1), log_pi_i

    def update(self, buffer, batch_size):
        if len(buffer) < batch_size:
            return

        # 训练批次
        obs, state, action, reward, next_obs, next_state, done = buffer.sample(batch_size)
        # 独立的 meta 批次，用于评估前瞻更新后的策略性能
        m_obs, m_state, _, _, _, _, _ = buffer.sample(batch_size)

        bsz = obs.shape[0]

        # ==========================================================
        # 步骤 1: 预计算所有智能体的 Next Action (更新 Critic 用)
        # ==========================================================
        with torch.no_grad():
            next_actions = []
            next_log_probs = []
            for j, agent in enumerate(self.agents):
                a_next, log_pi_next = agent.actor.sample(next_obs[:, j, :])
                next_actions.append(a_next)
                next_log_probs.append(log_pi_next)
            next_actions_cat = torch.cat(next_actions, dim=1)

        # ==========================================================
        # 步骤 2: 更新每个智能体的 Critic
        # ==========================================================
        for i, agent in enumerate(self.agents):
            with torch.no_grad():
                q1_next = agent.target_critic1(next_state, next_actions_cat)
                q2_next = agent.target_critic2(next_state, next_actions_cat)
                q_next_min = torch.min(q1_next, q2_next) - agent.alpha * next_log_probs[i]
                target_q = reward[:, i].unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * q_next_min

            action_cat = action.view(bsz, -1)
            q1_curr = agent.critic1(state, action_cat)
            q2_curr = agent.critic2(state, action_cat)

            critic1_loss = F.mse_loss(q1_curr, target_q)
            agent.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            agent.critic1_optimizer.step()

            critic2_loss = F.mse_loss(q2_curr, target_q)
            agent.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            agent.critic2_optimizer.step()

        # ==========================================================
        # 步骤 3: Meta-SAC 更新 —— 先用元梯度调 Alpha, 再更新 Actor
        # ==========================================================
        for i, agent in enumerate(self.agents):
            # ---------- (a) 内层前瞻: 用当前 alpha 对策略做一步可微更新 ----------
            actor_params = {n: p for n, p in agent.actor.named_parameters()}

            curr_cat, log_pi_i = self._joint_action_for_agent(i, obs, use_params=actor_params)
            q1_pi = agent.critic1(state, curr_cat)
            q2_pi = agent.critic2(state, curr_cat)
            q_pi_min = torch.min(q1_pi, q2_pi)

            alpha = agent.log_alpha.exp()  # 保留梯度, 使 theta' 依赖 alpha
            inner_actor_loss = (alpha * log_pi_i - q_pi_min).mean()

            grads = torch.autograd.grad(
                inner_actor_loss, list(actor_params.values()), create_graph=True
            )
            updated_params = {
                n: p - agent.inner_lr * g
                for (n, p), g in zip(actor_params.items(), grads)
            }

            # ---------- (b) 元目标: 在 meta batch 上评估前瞻策略的性能 ----------
            m_cat, _ = self._joint_action_for_agent(i, m_obs, use_params=updated_params)
            q1_meta = agent.critic1(m_state, m_cat)
            q2_meta = agent.critic2(m_state, m_cat)
            q_meta_min = torch.min(q1_meta, q2_meta)
            # 元损失不含熵项: 直接最大化前瞻策略的 Q 值
            meta_loss = -q_meta_min.mean()

            agent.alpha_optimizer.zero_grad()
            meta_loss.backward()   # 梯度经 updated_params 反传到 log_alpha
            agent.alpha_optimizer.step()

            # 更新显式 alpha 供后续使用
            agent.alpha = agent.log_alpha.exp().item()

            # ---------- (c) 用更新后的 alpha 做真正的 Actor 更新 ----------
            curr_cat2, log_pi_i2 = self._joint_action_for_agent(i, obs, use_params=None)
            q1_pi2 = agent.critic1(state, curr_cat2)
            q2_pi2 = agent.critic2(state, curr_cat2)
            q_pi_min2 = torch.min(q1_pi2, q2_pi2)

            actor_loss = (agent.alpha * log_pi_i2 - q_pi_min2).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

        # ==========================================================
        # 步骤 4: 软更新所有 Target 网络
        # ==========================================================
        self._soft_update_targets()

    def _soft_update_targets(self):
        for agent in self.agents:
            for target_param, param in zip(agent.target_critic1.parameters(), agent.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(agent.target_critic2.parameters(), agent.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
