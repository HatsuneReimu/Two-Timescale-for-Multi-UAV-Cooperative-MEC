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


class SACActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # 输出动作均值和动作对数标准差
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
        std = log_std.exp()
        normal = Normal(mu, std)

        # 重参数化采样 (rsample)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)  # 将动作压缩至 [-1, 1]
        action = y_t

        # 计算 log probability (修正 tanh 的雅可比行列式影响)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim, total_action_dim):
        super(SACCritic, self).__init__()
        # 集中式 Critic
        self.fc1 = nn.Linear(state_dim + total_action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# 3. 经验回放池
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
# 4. MASAC 核心算法实现
# ==========================================
class MASACAgent:
    def __init__(self, obs_dim, act_dim, state_dim, total_act_dim, lr=3e-4):
        # 1. 初始化 Actor
        self.actor = SACActor(obs_dim, act_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # 2. 初始化双 Critic 及其 Target 网络以防过拟合
        self.critic1 = SACCritic(state_dim, total_act_dim).to(device)
        self.critic2 = SACCritic(state_dim, total_act_dim).to(device)
        self.target_critic1 = SACCritic(state_dim, total_act_dim).to(device)
        self.target_critic2 = SACCritic(state_dim, total_act_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # 3. 自动调整温度参数 Alpha (熵的系数)
        self.target_entropy = -act_dim  # 目标熵值，通常设为 -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()


class MASAC:
    def __init__(self, env, gamma=0.95, tau=0.01, lr=3e-4):
        self.num_agents = env.num_uavs
        self.gamma = gamma
        self.tau = tau
        obs_dim = 3 + 2 * env.num_users  # 用户位置 + 所有UAV位置
        action_dim = 3
        state_dim = 3 * env.num_uavs + 2 * env.num_users  # 用户位置 + 所有UAV位置
        total_act_dim = env.num_uavs * action_dim

        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(
                MASACAgent(obs_dim, action_dim, state_dim, total_act_dim, lr=lr)
            )

    def select_action(self, obs, evaluate=False):

        # evaluate=False 训练时(随机采样)
        # evaluate=True 测试时(直接取均值)

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

    def update(self, buffer, batch_size):
        if len(buffer) < batch_size:
            return

        obs, state, action, reward, next_obs, next_state, done = buffer.sample(batch_size)

        # ==========================================================
        # 步骤 1: 预计算所有智能体的 Next Action 用于更新 Critic
        # ==========================================================
        with torch.no_grad():
            next_actions = []
            next_log_probs = []
            for j, agent in enumerate(self.agents):
                a_next, log_pi_next = agent.actor.sample(next_obs[:, j, :])
                next_actions.append(a_next)
                next_log_probs.append(log_pi_next)
            next_actions_cat = torch.cat(next_actions, dim=1)  # (Batch, Total_Act_Dim)

        # ==========================================================
        # 步骤 2: 更新每个智能体的 Critic
        # ==========================================================
        for i, agent in enumerate(self.agents):
            # --- 计算 Target Q ---
            with torch.no_grad():
                q1_next = agent.target_critic1(next_state, next_actions_cat)
                q2_next = agent.target_critic2(next_state, next_actions_cat)
                # 取两个 Critic 中的最小值 (缓解过估) + 熵的奖励
                q_next_min = torch.min(q1_next, q2_next) - agent.alpha * next_log_probs[i]
                target_q = reward[:, i].unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * q_next_min

            # --- 计算 Current Q ---
            action_cat = action.view(batch_size, -1)
            q1_curr = agent.critic1(state, action_cat)
            q2_curr = agent.critic2(state, action_cat)

            # --- 更新 Critic ---
            critic1_loss = F.mse_loss(q1_curr, target_q)
            agent.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            agent.critic1_optimizer.step()

            critic2_loss = F.mse_loss(q2_curr, target_q)
            agent.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            agent.critic2_optimizer.step()

        # ==========================================================
        # 步骤 3: 更新每个智能体的 Actor 和 Alpha(温度系数)
        # ==========================================================
        for i, agent in enumerate(self.agents):
            # 获取所有智能体在当前状态下生成的**最新**动作 (使用当前策略)
            curr_actions = []
            log_pi_curr = None
            for j, ag in enumerate(self.agents):
                if j == i:
                    a_curr, log_pi = ag.actor.sample(obs[:, j, :])
                    curr_actions.append(a_curr)
                    log_pi_curr = log_pi
                else:
                    # 其他智能体的动作不产生梯度，直接detach
                    with torch.no_grad():
                        a_curr, _ = ag.actor.sample(obs[:, j, :])
                    curr_actions.append(a_curr)
            curr_actions_cat = torch.cat(curr_actions, dim=1)

            # --- 更新 Actor ---
            # 最小化: α * log(π) - Q
            q1_pi = agent.critic1(state, curr_actions_cat)
            q2_pi = agent.critic2(state, curr_actions_cat)
            q_pi_min = torch.min(q1_pi, q2_pi)

            actor_loss = (agent.alpha * log_pi_curr - q_pi_min).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # --- 更新 Alpha ---
            # Alpha Loss: - α * (log(π) + H_target)
            alpha_loss = -(agent.log_alpha * (log_pi_curr + agent.target_entropy).detach()).mean()

            agent.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            agent.alpha_optimizer.step()

            # 更新显式的 alpha 供下次循环使用
            agent.alpha = agent.log_alpha.exp().item()

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

