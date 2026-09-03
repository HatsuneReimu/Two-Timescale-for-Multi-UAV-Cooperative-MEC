import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 将动作限制在[-1, 1] 内


class Critic(nn.Module):
    def __init__(self, state_dim, total_action_dim):
        super(Critic, self).__init__()
        # 集中式 Critic 的输入是 全局状态 + 所有智能体的动作
        self.fc1 = nn.Linear(state_dim + total_action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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

class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, state_dim, total_act_dim, lr_actor, lr_critic):
        # 初始化 Actor 和 Critic 网络
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, total_act_dim).to(device)
        self.target_critic = Critic(state_dim, total_act_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)


class MADDPG:
    def __init__(self, env, gamma=0.95, tau=0.01, lr_a=1e-4, lr_c=1e-3):
        self.num_agents = env.num_uavs
        self.gamma = gamma
        self.tau = tau
        # obs_dim = 3 * env.num_uavs+ 2 * env.num_users
        obs_dim = 3 + 2 * env.num_users  # 用户位置 + 所有UAV位置
        action_dim = 3
        state_dim = 3 * env.num_uavs+ 2 * env.num_users

        total_act_dim = env.num_uavs * 3

        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(
                MADDPGAgent(obs_dim, action_dim, state_dim, total_act_dim, lr_a, lr_c)
            )

    def select_action(self, obs, noise=0.0):
        actions = []
        for i, agent in enumerate(self.agents):
            o = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                a = agent.actor(o).cpu().numpy().flatten()
            # 增加高斯噪声进行探索
            if noise > 0:
                a += np.random.normal(0, noise, size=a.shape)
            actions.append(np.clip(a, -1, 1))
        return np.array(actions)

    def update(self, buffer, batch_size):
        if len(buffer) < batch_size:
            return

        obs, state, action, reward, next_obs, next_state, done = buffer.sample(batch_size)

        for i, agent in enumerate(self.agents):
            # ---------------- 训练 Critic ----------------
            with torch.no_grad():
                # 计算所有智能体的下一个 Target 动作
                next_actions = []
                for j, ag in enumerate(self.agents):
                    next_actions.append(ag.target_actor(next_obs[:, j, :]))
                next_actions_cat = torch.cat(next_actions, dim=1)  # Shape: (Batch, Total_Act_Dim)

                # 计算 Target Q 值
                target_q = agent.target_critic(next_state, next_actions_cat).squeeze(1)
                # target_q = reward[:, i] + self.gamma * target_q * (1 - done) # 如果环境有done可以使用
                y_expected = reward[:, i] + self.gamma * target_q

            # 当前动作铺平
            action_cat = action.view(batch_size, -1)
            q_value = agent.critic(state, action_cat).squeeze(1)

            critic_loss = F.mse_loss(q_value, y_expected)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            # ---------------- 训练 Actor ----------------
            # 将当前网络产生的动作和其它智能体（detach）产生的动作拼接起来，计算全局 Q 值
            curr_actions = []
            for j, ag in enumerate(self.agents):
                if j == i:
                    curr_actions.append(ag.actor(obs[:, j, :]))  # 当前智能体使用自己的 Actor 梯度
                else:
                    curr_actions.append(ag.actor(obs[:, j, :]).detach())  # 其它智能体的动作截断梯度

            curr_actions_cat = torch.cat(curr_actions, dim=1)
            actor_loss = -agent.critic(state, curr_actions_cat).mean()  # 最大化 Q 值

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor_optimizer.step()

        # ---------------- 软更新 Target 网络 ----------------
        self._soft_update_targets()

    def _soft_update_targets(self):
        for agent in self.agents:
            for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)