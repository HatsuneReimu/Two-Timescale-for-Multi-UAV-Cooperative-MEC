import argparse
import numpy as np
import torch
import time
from CAMRL.runner.w1_w2.ENV_DHHO import ENV
from CAMRL.algorithms.MAPPO_Lagr import MAPPO
import wandb
import matplotlib.pyplot as plt

# -------------------------- 主程序 --------------------------
def get_args():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument("--algorithm_name", type=str,
                        default=' ', choices=[ "mappo_lagr"])

    parser.add_argument('--num_users', type=int, default=20)
    parser.add_argument('--num_uavs', type=int, default=5)
    parser.add_argument('--area_x', type=float, default=500.0)
    parser.add_argument('--area_y', type=float, default=500.0)
    parser.add_argument('--max_height', type=float, default=80)
    parser.add_argument('--min_height', type=float, default=50)
    parser.add_argument('--uav_max_speed', type=float, default=50.0)  # UAV最大速度 (m/s)
    parser.add_argument('--small_time_slot', type=float, default=1.5)

    # 用户与任务
    parser.add_argument('--user_computing_capacity', type=float, default=1e8)
    parser.add_argument('--user_tx_power', type=float, default=0.8)
    parser.add_argument('--task_size_min', type=float, default=3.2e7)
    parser.add_argument('--task_size_max', type=float, default=3.2e7)
    parser.add_argument('--task_complexity_min', type=float, default=50.0)
    parser.add_argument('--task_complexity_max', type=float, default=50.0)

    # UAV参数
    parser.add_argument('--uav_tx_power', type=float, default=5.0)
    parser.add_argument('--uav_max_computing_capacity', type=float, default=5e9)

    # 通信参数
    parser.add_argument('--channel_bandwidth', type=float, default=20e6)
    parser.add_argument('--noise_power', type=float, default=-174)
    parser.add_argument('--eta_los', type=float, default=0.1)
    parser.add_argument('--eta_nlos', type=float, default=21)
    parser.add_argument('--light_speed', type=float, default=3e8)
    parser.add_argument('--carrier_frequency_g2a', type=float, default=2.4e9)
    parser.add_argument('--carrier_frequency_a2a', type=float, default=5.8e9)

    # 训练参数
    parser.add_argument('--policy_clip', type=float, default=0.2)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--max_episodes', type=int, default=500)

    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--std_x_coef", type=float, default=1)
    parser.add_argument("--std_y_coef", type=float, default=0.5)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--mini_batch_size", type=int, default=32,
                        help='mini batch size for ppo update')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    # todo: lagrangian_coef is the lagrangian coefficient for mappo_lagrangian
    parser.add_argument("--lamda_lagr", type=float, default=0.78,
                        help='lagrangrian coef coefficient (default: 0.78)')
    parser.add_argument("--lagrangian_coef_rate", type=float, default=5e-4,
                        help='lagrangrian coef learning rate (default: 5e-4)')

    parser.add_argument("--lagrangian_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=False,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--use_popart", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--episode_length", type=int,
                        default=128, help="Max length for any episode")

    parser.add_argument("--safety_bound", type=float, default=0.1, help="constraint upper bound")


    return parser.parse_args()

def log_train(train_infos, total_num_steps):
    for agent_id in range(len(train_infos)):
        for k, v in train_infos[agent_id].items():
            agent_k = "agent%i/" % agent_id + k
            wandb.log({agent_k: v}, step=total_num_steps)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 初始化环境
    env = ENV(args)

    # 计算观测/状态/动作维度
    obs_dim = 3 + 2 * args.num_users
    state_dim = 3 * args.num_uavs + 2 * args.num_users   # 用户位置 + 所有UAV位置
    action_dim = 3  # x/y方向速度

    # 修正变量名：num_agents对应num_uavs
    args.num_agents = args.num_uavs
    agents = [MAPPO(args, obs_dim, state_dim, action_dim, device) for _ in range(args.num_uavs)]

    start = time.time()
    wandb.init(
        project="w1_w2",  # 项目名保留，仅作为本地标识
        entity="assassin021012-nanjing-university-of-information-science",  # 可保留（离线时仅本地记录）
        config=args,  # 保留：本地记录实验参数
    )

    env.reset()
    print("User positions:", env.user_positions)

    pos_array = np.array(env.user_positions)
    x_coords = pos_array[:, 0]  # 提取所有点的x坐标（第0列）
    y_coords = pos_array[:, 1]  # 提取所有点的y坐标（第1列）

    pos_array1 = np.array(env.uav_positions)
    x_coords1 = pos_array1[:, 0]  # 提取所有点的x坐标（第0列）
    y_coords1 = pos_array1[:, 1]  # 提取所有点的y坐标（第1列）

    # 2. 设置图表样式，绘制黑点
    plt.figure(figsize=(10, 10))  # 创建8×8英寸的正方形画布（保证坐标比例不失真）
    plt.scatter(x_coords, y_coords, color='red', s=25, label='User')  # 绘制散点，color='black'指定黑点，s=20设置点的大小
    plt.scatter(x_coords1, y_coords1, color='blue', s=25, label='UAV')  # 绘制UAV散点

    # 3. 配置坐标轴（匹配500×500的范围）
    plt.xlim(0, 500)  # x轴范围0到500
    plt.ylim(0, 500)  # y轴范围0到500
    plt.xlabel('X Coordinate')  # x轴标签
    plt.ylabel('Y Coordinate')  # y轴标签
    plt.title('20 Random Points in [0, 500]×[0, 500]')  # 图表标题
    plt.grid(True, alpha=0.3)  # 显示网格（alpha=0.3设置透明度，不遮挡黑点）

    # 4. 显示图表
    plt.show()

    # 训练循环
    for i_episode in range(args.max_episodes):
        # 学习率衰减（修正变量引用错误）
        for agent_idx in range(args.num_uavs):
            agents[agent_idx].policy.lr_decay(i_episode, args.max_episodes)

        obs, state = env.reset()
        episodes_rewards = [0.0 for _ in range(args.num_uavs)]
        episodes_costs = [0.0 for _ in range(args.num_uavs)]
        episodes_energys = [0.0 for _ in range(args.num_uavs)]
        episodes_delays = [0.0 for _ in range(args.num_uavs)]

        # Buffer 初始化：将初始观测填入 step=0
        for agent_idx in range(args.num_uavs):
            agents[agent_idx].buffer.share_obs[0] = state.copy()
            agents[agent_idx].buffer.obs[0] = obs[agent_idx].copy()

        for t in range(args.episode_length):
            # 临时列表收集数据
            values = []
            costs_preds = []
            actions = []
            action_log_probs = []

            with torch.no_grad():
                for agent_idx in range(args.num_uavs):
                    # 转换数据为 tensor
                    p_obs = torch.tensor(obs[agent_idx], dtype=torch.float).to(device).unsqueeze(0)
                    p_share_obs = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0)

                    value, cost_preds, action, action_log_prob = agents[agent_idx].policy.get_actions(
                        p_share_obs, p_obs, deterministic=False
                    )

                    values.append(value.detach().cpu().numpy())
                    costs_preds.append(cost_preds.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    action_log_probs.append(action_log_prob.detach().cpu().numpy())

            env_actions = np.array([a[0] for a in actions])

            next_obs, next_state, rewards, costs, delay, energy, done = env.step(env_actions)

            for agent_idx in range(args.num_uavs):
                episodes_rewards[agent_idx] += rewards[agent_idx]
                episodes_costs[agent_idx] += costs[agent_idx]
                episodes_energys[agent_idx] += energy[agent_idx]
                episodes_delays[agent_idx] += delay[agent_idx]

            for agent_idx in range(args.num_uavs):
                agents[agent_idx].buffer.insert(
                    next_state,  # 存入 t+1 的共享状态
                    next_obs[agent_idx],  # 存入 t+1 的个体观测
                    actions[agent_idx][0],  # 存入 t 的动作
                    action_log_probs[agent_idx][0],  # 存入 t 的动作对数概率
                    values[agent_idx][0],  # 存入 t 的价值预测
                    np.array([rewards[agent_idx]]),
                    np.array([costs[agent_idx]]),
                    costs_preds[agent_idx][0]
                )

            state = next_state
            obs = next_obs

        train_infos = []
        factor = np.ones((args.episode_length, action_dim), dtype=np.float32)
        for agent_idx in torch.randperm(args.num_uavs):
            agents[agent_idx].buffer.update_factor(factor)
            agents[agent_idx].buffer.return_aver_insert(agents[agent_idx].buffer.costs)
            train_info = agents[agent_idx].train()
            train_infos.append(train_info)

        total_num_steps = (i_episode + 1)
        if (i_episode + 1) % 1 == 0:
            end = time.time()
            print("\n Updates {}/{} episodes, total num timesteps {}, FPS {}.\n"
                  .format(
                          i_episode + 1,
                          args.max_episodes,
                          total_num_steps,
                          int(total_num_steps / (end - start))))
            log_train(train_infos, total_num_steps)

            total_reward_log_dict = {}
            total_reward_key = f"episode_total_reward"
            total_reward_log_dict[total_reward_key] = sum(episodes_rewards)
            # 保持step一致，确保奖励和其他训练指标对齐
            wandb.log(total_reward_log_dict, step=total_num_steps)

            cost_log_dict = {}
            cost_key = f"episode_total_cost"
            cost_log_dict[cost_key] = sum(episodes_costs)
            # 保持step一致，确保奖励和其他训练指标对齐
            wandb.log(cost_log_dict, step=total_num_steps)

            delay_log_dict = {}
            delay_key = f"episode_total_delay"
            delay_log_dict[delay_key] = sum(episodes_delays)
            # 保持step一致，确保奖励和其他训练指标对齐
            wandb.log(delay_log_dict, step=total_num_steps)

            energy_log_dict = {}
            energy_key = f"episode_total_energy"
            energy_log_dict[energy_key] = sum(episodes_energys)
            # 保持step一致，确保奖励和其他训练指标对齐
            wandb.log(energy_log_dict, step=total_num_steps)

    wandb.finish()

if __name__ == '__main__':
    main()
