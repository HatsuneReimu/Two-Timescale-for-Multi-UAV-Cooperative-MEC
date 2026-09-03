"""
修改：
在ENV2的基础上增加了无人机飞行的能耗损失 和 惩罚项
"""


import numpy as np
from scipy.spatial.distance import cdist
import random

class GWOSolver:
    def __init__(self, env, args, failure_status):
        self.env = env
        self.failure_status = failure_status
        self.active_uavs = np.where(failure_status == 0)[0]
        self.T_SMALL = args.small_time_slot

        # 预计算速率 (Rate Matrices)
        # print(">>> [GWO] 正在预计算信道状态...")
        pl_g2a = env.calculate_a2g_path_loss()
        self.rate_g2a = env.calculate_transmission_rate(pl_g2a, env.user_tx_power)
        pl_a2a = env.calculate_a2a_path_loss()
        self.rate_a2a = env.calculate_transmission_rate(pl_a2a, env.uav_tx_power)

        # 预计算任务的计算量 W 和 sqrt(W)
        self.W_vec = self.env.users_task_size * self.env.users_task_complexity
        self.sqrt_W_vec = np.sqrt(self.W_vec)

    def heuristic_repair(self, uav_indices):
        """
        启发式修复机制：
        如果发现用户违反时延约束，尝试将其移动到能满足约束且时延最小的 UAV。
        """
        # 最多尝试 2 轮修复，避免死循环
        for _ in range(2):
            # 1. 计算当前所有 UAV 的负载 (sum_sqrt_W)
            uav_load_sum_sqrt = np.zeros(self.env.num_uavs)
            for m, u_idx in enumerate(uav_indices):
                if u_idx != -1:
                    uav_load_sum_sqrt[u_idx] += self.sqrt_W_vec[m]

            # 2. 检查每个用户的时延
            violators = []
            current_latencies = np.zeros(self.env.num_users)

            for m in range(self.env.num_users):
                u = uav_indices[m]
                if u == -1:
                    violators.append(m)
                    continue

                # 计算时延
                lat = self._calculate_single_user_latency(m, u, uav_load_sum_sqrt[u])
                current_latencies[m] = lat
                if lat > self.T_SMALL:
                    violators.append(m)

            if not violators:
                break  # 所有约束满足

            # 3. 尝试修复违规用户
            for m in violators:
                best_uav = uav_indices[m]
                min_lat = current_latencies[m] if best_uav != -1 else float('inf')
                found_feasible = False if min_lat > self.T_SMALL else True

                # 遍历所有存活 UAV
                for u_cand in self.active_uavs:
                    if u_cand == uav_indices[m]:
                        continue

                    # 估算移动后的新负载：
                    # 目标 UAV 增加该用户负载
                    new_load_cand = uav_load_sum_sqrt[u_cand] + self.sqrt_W_vec[m]

                    # 估算该用户在目标 UAV 上的新时延
                    est_lat = self._calculate_single_user_latency(m, u_cand, new_load_cand)

                    # 贪婪选择：优先选择满足约束的；若都满足，选最小的
                    if est_lat <= self.T_SMALL:
                        if not found_feasible or est_lat < min_lat:
                            best_uav = u_cand
                            min_lat = est_lat
                            found_feasible = True
                    elif not found_feasible and est_lat < min_lat:
                        # 如果都无法满足，选择违规程度最小的
                        best_uav = u_cand
                        min_lat = est_lat

                # 更新连接
                if best_uav != -1 and best_uav != uav_indices[m]:
                    # 更新全局负载记录，以便后续循环使用最新状态
                    old_u = uav_indices[m]
                    if old_u != -1:
                        uav_load_sum_sqrt[old_u] -= self.sqrt_W_vec[m]
                    uav_indices[m] = best_uav
                    uav_load_sum_sqrt[best_uav] += self.sqrt_W_vec[m]

        return uav_indices

    def _calculate_single_user_latency(self, m, u, uav_sum_sqrt_W):
        """辅助函数：计算单个用户 m 在 UAV u 上特定负载下的总时延"""
        # 1. Access
        acc_uav = self.env.user_uav_connections[m]
        if acc_uav == -1: return 1e6  # 无连接惩罚

        r_up = self.rate_g2a[m, acc_uav]
        if r_up < 1e-5: return 1e6
        t_up = self.env.users_task_size[m] / r_up

        # 2. Trans
        t_trans = 0
        if acc_uav != u:
            r_tr = self.rate_a2a[acc_uav, u]
            if r_tr < 1e-5: return 1e6
            t_trans = self.env.users_task_size[m] / r_tr

        # 3. Comp (基于闭式解反推时延)
        # T_comp = W_m / f_m, 且 f_m = (sqrt(W_m) / Sum_sqrt_W) * F_max
        # 代入得 T_comp = (sqrt(W_m) * Sum_sqrt_W) / F_max
        t_comp = (self.sqrt_W_vec[m] * uav_sum_sqrt_W) / self.env.uav_max_computing_capacity

        return t_up + t_trans + t_comp

    def calculate_objective(self, positions_continuous):
        """计算适应度并返回详细信息"""
        # 1. 离散化
        uav_indices = np.round(positions_continuous).astype(int)
        uav_indices = self._correct_positions(uav_indices)

        # 2. [关键修改] 调用启发式修复，尝试强制满足约束
        uav_indices = self.heuristic_repair(uav_indices)

        # 3. 计算最终时延与惩罚
        total_lat = 0
        penalty = 0
        user_lats = np.zeros(self.env.num_users)
        allocation_scheme = {}

        # 统计负载
        uav_tasks = {u: [] for u in range(self.env.num_uavs)}
        for m, u in enumerate(uav_indices):
            if u != -1: uav_tasks[u].append(m)

        for u in range(self.env.num_uavs):
            users = uav_tasks[u]
            if not users:
                allocation_scheme[u] = []
                continue

            # 局部负载计算
            sum_sqrt = 0
            for m in users: sum_sqrt += self.sqrt_W_vec[m]

            # 资源分配
            f_allocs = []
            if sum_sqrt > 0:
                for m in users:
                    f = (self.sqrt_W_vec[m] / sum_sqrt) * self.env.uav_max_computing_capacity
                    f_allocs.append(f)
            else:
                f_allocs = [0] * len(users)

            allocation_scheme[u] = f_allocs

            # 计算最终时延
            for i, m in enumerate(users):
                lat = self._calculate_single_user_latency(m, u, sum_sqrt)
                user_lats[m] = lat

                # 约束检查
                if lat > self.T_SMALL:
                    # 极大的惩罚，确保即便修复失败，该解也被淘汰
                    penalty += 1e9 * (lat - self.T_SMALL) + 1e5

        fitness = np.sum(user_lats) + penalty
        return fitness, np.sum(user_lats), uav_indices, user_lats, allocation_scheme

    def _correct_positions(self, indices):
        indices = np.clip(indices, 0, self.env.num_uavs - 1)
        for i, u_idx in enumerate(indices):
            if self.failure_status[u_idx] == 1:
                if len(self.active_uavs) > 0:
                    indices[i] = random.choice(self.active_uavs)
                else:
                    indices[i] = -1
        return indices

    def solve(self, pop_size=30, max_iter=100):
        if len(self.active_uavs) == 0:
            print("所有 UAV 失效。")
            return None

        dim = self.env.num_users

        # 初始化狼群位置
        positions = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for d in range(dim):
                positions[i, d] = random.choice(self.active_uavs)
                positions[i, d] += np.random.uniform(-0.49, 0.49)

        alpha_pos = np.zeros(dim);
        alpha_score = float("inf")
        beta_pos = np.zeros(dim);
        beta_score = float("inf")
        delta_pos = np.zeros(dim);
        delta_score = float("inf")

        best_details = None
        convergence_curve = []

        # print(f">>> GWO (Enhanced) 开始迭代 (Pop={pop_size}, Iter={max_iter})...")

        for t in range(max_iter):
            for i in range(pop_size):
                # 边界约束
                positions[i, :] = np.clip(positions[i, :], 0, self.env.num_uavs - 1)

                # 评估（内部包含 repair）
                fit, true_lat, discrete_sol, u_lats, scheme = self.calculate_objective(positions[i, :])

                # 更新 Alpha, Beta, Delta
                if fit < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = fit, positions[i, :].copy()  # 注意：这里保存的是 repair 前的连续位置，或者应保存 repair 后的？
                    # GWO 标准流程通常保存连续位置。
                    # 但为了输出最优解，我们保存 discrete_sol 到 best_details
                    best_details = (true_lat, discrete_sol, u_lats, scheme)
                elif fit < beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = fit, positions[i, :].copy()
                elif fit < delta_score:
                    delta_score, delta_pos = fit, positions[i, :].copy()

            convergence_curve.append(alpha_score)

            # GWO 位置更新
            a = 2 - t * (2 / max_iter)
            for i in range(pop_size):
                for d in range(dim):
                    r1, r2 = random.random(), random.random()
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * alpha_pos[d] - positions[i, d])
                    X1 = alpha_pos[d] - A1 * D_alpha

                    r1, r2 = random.random(), random.random()
                    A2, C2 = 2 * a * r1 - a, 2 * r2
                    D_beta = abs(C2 * beta_pos[d] - positions[i, d])
                    X2 = beta_pos[d] - A2 * D_beta

                    r1, r2 = random.random(), random.random()
                    A3, C3 = 2 * a * r1 - a, 2 * r2
                    D_delta = abs(C3 * delta_pos[d] - positions[i, d])
                    X3 = delta_pos[d] - A3 * D_delta

                    positions[i, d] = (X1 + X2 + X3) / 3.0

        return best_details, convergence_curve

class ENV:
    def __init__(self, args):
        self.args = args
        # -------------------------- 1. 基础配置参数 --------------------------
        self.num_users = args.num_users
        self.num_uavs = args.num_uavs
        self.area_size = np.array([args.area_x, args.area_y])
        self.uav_max_speed = args.uav_max_speed  # UAV最大速度 (m/s)

        # -------------------------- 2. 用户相关参数 --------------------------
        self.task_size_range = (args.task_size_min, args.task_size_max)
        self.task_complexity_range = (args.task_complexity_min, args.task_complexity_max)
        self.user_computing_capacity = args.user_computing_capacity
        self.user_tx_power = args.user_tx_power
        self.user_positions = np.zeros((self.num_users, 2))
        # 新增：存储当前任务的size和complexity（供GWOSolver使用）
        self.users_task_size = np.zeros(self.num_users)
        self.users_task_complexity = np.zeros(self.num_users)

        # -------------------------- 3. 连接与任务分配相关参数 --------------------------
        self.user_uav_distances = np.full((self.num_users, self.num_uavs), np.inf)

        # -------------------------- 4. UAV相关参数 --------------------------
        # 移除固定高度，改为每个UAV的三维位置（x,y,z）
        self.uav_positions = np.zeros((self.num_uavs, 3))  # UAV三维位置 (num_uavs, 3)
        self.uav_tx_power = args.uav_tx_power
        self.uav_max_computing_capacity = args.uav_max_computing_capacity
        self.uav_height_min =args.min_height  # UAV最小高度
        self.uav_height_max = args.max_height  # UAV最大高度
        # -------------------------- 5. 通信信道参数 --------------------------
        self.channel_bandwidth = args.channel_bandwidth
        self.noise_power = args.noise_power
        self.eta_los = args.eta_los
        self.eta_nlos = args.eta_nlos
        self.light_speed = args.light_speed
        self.carrier_frequency_g2a = args.carrier_frequency_g2a
        self.carrier_frequency_a2a = args.carrier_frequency_a2a
        self.path_loss_exponent = 2.2
        self.epsilon_0 = 1  # 噪声的标准差，方差为 epsilon_0^2

        # ==================== 新增：能耗模型参数 ====================
        # 水平飞行功率参数（默认值参考无人机能耗经典取值，可根据需求调整）
        self.P0 = 80.0  # 悬停时叶片轮廓功率 (W)
        self.P1 = 100.0  # 悬停时诱导功率 (W)
        self.U_tips = 120.0  # 旋翼叶片尖端速度 (m/s)
        self.v0 = 4.0  # 平均旋翼诱导速度 (m/s)
        self.d0 = 0.6  # 机身阻力比
        self.rho0 = 1.225  # 空气密度 (kg/m³)
        self.s0 = 0.08  # 旋翼实度
        self.A0 = 0.5  # 旋翼盘面积 (m²)
        # 垂直飞行功率参数
        self.W = 2.0  # 无人机质量 (kg)
        self.g = 9.81  # 重力加速度 (m/s²)
        # 能耗惩罚系数（控制能耗在奖励中的权重，可调整）
        self.energy_penalty_coeff = 0.005

        self.random_seed = 123

    def _set_random_seed(self):
        """统一设置随机数种子，确保结果可复现"""
        # 设置NumPy随机种子（核心，因为代码中用了np.random）
        np.random.seed(self.random_seed)
        # 可选：如果代码中用到Python内置random，也需要设置
        random.seed(self.random_seed)

    def simulate_uav_failure(self):
        failure_array = np.random.binomial(1, 0.01, self.num_uavs)

        if np.all(failure_array == 1):
            random_idx = np.random.randint(0, self.num_uavs)
            failure_array[random_idx] = 0

        return failure_array


    def _generate_user_positions(self):
        """生成所有用户的固定水平位置"""
        user_positions = []
        existing_positions = set()
        max_attempts = self.num_users * 5
        attempts = 0

        while len(user_positions) < self.num_users and attempts < max_attempts:
            x = round(np.random.uniform(low=0.0, high=self.area_size[0]), 2)
            y = round(np.random.uniform(low=0.0, high=self.area_size[1]), 2)
            pos_key = (x, y)
            if pos_key not in existing_positions:
                existing_positions.add(pos_key)
                user_positions.append([x, y])
            attempts += 1

        while len(user_positions) < self.num_users:
            user_positions.append([np.random.uniform(0, self.area_size[0]),
                                   np.random.uniform(0, self.area_size[1])])

        return np.array(user_positions, dtype=np.float64)


    def _generate_uav_position(self):
        """生成所有无人机的初始水平位置"""
        uav_positions = []
        existing_positions = set()
        max_attempts = self.num_uavs * 5
        attempts = 0

        while len(uav_positions) < self.num_uavs and attempts < max_attempts:
            x = round(np.random.uniform(low=0.0, high=self.area_size[0]), 2)
            y = round(np.random.uniform(low=0.0, high=self.area_size[1]), 2)
            z = round(np.random.uniform(low=self.uav_height_min, high=self.uav_height_max), 2)
            pos_key = (x, y, z)
            if pos_key not in existing_positions:
                existing_positions.add(pos_key)
                uav_positions.append([x, y, z])
            attempts += 1

        # 若尝试次数用尽仍未生成足够位置，直接补充（不检查唯一性）
        while len(uav_positions) < self.num_uavs:
            x = np.random.uniform(0, self.area_size[0])
            y = np.random.uniform(0, self.area_size[1])
            z = np.random.uniform(self.uav_height_min, self.uav_height_max)
            uav_positions.append([x, y, z])

        return np.array(uav_positions, dtype=np.float64)


    def get_user_all_uav_distances(self):
        """计算用户（z=0）与UAV的3D距离"""
        # 用户补全z=0维度，变为三维坐标
        user_pos_3d = np.hstack([self.user_positions, np.zeros((self.num_users, 1))])
        # 计算三维欧氏距离
        user_uav_distances = cdist(user_pos_3d, self.uav_positions, metric='euclidean').round(2)
        return user_uav_distances


    def calculate_los_probability(self):
        """计算视距概率（适配每个UAV的独立高度）"""
        # 提取UAV的水平位置（x,y）
        uav_2d_positions = self.uav_positions[:, :2]
        # 计算用户与UAV的水平距离
        user_uav_level_distances = cdist(self.user_positions, uav_2d_positions, metric='euclidean').round(2)
        user_uav_level_distances = np.maximum(user_uav_level_distances, 1e-6)

        # 提取每个UAV的高度（z坐标），扩展维度以匹配水平距离矩阵
        uav_heights = self.uav_positions[:, 2].reshape(1, -1)  # (1, num_uavs)
        # 计算每个用户-无人机对的仰角（弧度）
        elevation_angle_rad = np.arctan(uav_heights / user_uav_level_distances)
        elevation_angle_deg = elevation_angle_rad * 180 / np.pi

        los_prob = 1 / (1 + 9.61 * np.exp(-0.168 * (elevation_angle_deg - 9.61)))
        return los_prob.round(4)


    def calculate_a2g_path_loss(self):
        """计算A2G路径损耗（适配每个UAV的独立高度）"""
        los_prob = self.calculate_los_probability()
        # 提取UAV水平位置，计算用户-UAV水平距离
        uav_2d_positions = self.uav_positions[:, :2]
        user_uav_level_distances = cdist(self.user_positions, uav_2d_positions, metric='euclidean').round(2)
        # 提取每个UAV的高度并扩展维度
        uav_heights = self.uav_positions[:, 2].reshape(1, -1)
        # 计算三维直线距离
        straight_distance = np.sqrt(np.square(user_uav_level_distances) + np.square(uav_heights))
        straight_distance = np.maximum(straight_distance, 1e-6)

        # 计算自由空间路径损耗
        fs_pl_term = 10 * np.log10(
            np.power((4 * np.pi * self.carrier_frequency_g2a * straight_distance) / self.light_speed,
                     self.path_loss_exponent))
        # 视距/非视距衰减
        los_nlos_attenuation = los_prob * self.eta_los + (1 - los_prob) * self.eta_nlos
        total_pl = fs_pl_term + los_nlos_attenuation
        return total_pl.round(2)


    def calculate_a2a_path_loss(self):
        """计算A2A路径损耗（适配UAV三维位置）"""
        # 计算UAV之间的三维欧氏距离
        uav_uav_distances = cdist(self.uav_positions, self.uav_positions, metric='euclidean').round(2)
        uav_uav_distances = np.maximum(uav_uav_distances, 1e-6)

        # 自由空间路径损耗计算
        fspi = 20 * np.log10((4 * np.pi * self.carrier_frequency_a2a * uav_uav_distances) / self.light_speed)
        np.fill_diagonal(fspi, 0.0)  # 自身距离损耗为0
        return fspi.round(2)


    def calculate_transmission_rate(self, pl_db, tx_power):
        """计算传输速率（逻辑不变，适配维度）"""
        pl_linear = np.power(10, pl_db / 10)
        pl_linear = np.maximum(pl_linear, 1e-6)

        noise_spectral_density_W_per_Hz = np.power(10, (self.noise_power - 30) / 10)
        sigma_square = noise_spectral_density_W_per_Hz * self.channel_bandwidth
        sigma_square = np.maximum(sigma_square, 1e-23)

        if len(pl_db.shape) == 2:
            power_matrix = np.full_like(pl_db, tx_power, dtype=np.float64)
        elif len(pl_db.shape) == 1:
            power_matrix = np.full_like(pl_db, tx_power, dtype=np.float64)
        else:
            power_matrix = np.full_like(pl_db, tx_power, dtype=np.float64)
            np.fill_diagonal(power_matrix, 0.0)

        snr = power_matrix / (pl_linear * sigma_square)
        snr = np.minimum(snr, 1e6)
        transmission_rate = self.channel_bandwidth * np.log2(1 + snr)

        transmission_rate = np.maximum(transmission_rate, 0)
        return transmission_rate.round(2)

    def update_user_uav_connections(self, failure_status):
        dists = self.user_uav_distances.copy()
        failed = np.where(failure_status == 1)[0]
        if len(failed) > 0: dists[:, failed] = np.inf
        self.user_uav_connections = np.argmin(dists, axis=1)
        self.user_uav_connections[np.min(dists, axis=1) == np.inf] = -1
        return self.user_uav_connections


    def _calculate_uav_energy_consumption(self, uav_velocity, flight_time):
        """
        计算单个无人机的飞行能耗（严格对应给定公式）
        参数：
            uav_velocity: 无人机三维速度向量 [vx, vy, vz] (m/s)
            flight_time: 飞行时间 (s)，对应公式中的T
        返回：
            energy: 该无人机的飞行能耗 (J，焦耳，1W*s=1J)
        """
        # 1. 拆分水平/垂直速度，并计算模（避免负数影响）
        v_xy = uav_velocity[:2]  # 水平速度分量 [vx, vy]
        v_z = uav_velocity[2]    # 垂直速度分量 vz
        v_xy_norm = np.linalg.norm(v_xy)  # 水平速度模 ||ϑ_{x,y}||
        v_z_abs = np.abs(v_z)             # 垂直速度模 ||v_z||

        # 2. 计算水平飞行功率 P_fly (W)
        # 第一部分：P0*(1 + 3*||v_xy||² / U_tips²)
        term1 = self.P0 * (1 + 3 * (v_xy_norm ** 2) / (self.U_tips ** 2))
        # 第二部分：P1*sqrt(1 + ||v_xy||⁴/(4*v0⁴) - ||v_xy||²/(2*v0²))
        inner_term = 1 + (v_xy_norm ** 4) / (4 * (self.v0 ** 4)) - (v_xy_norm ** 2) / (2 * (self.v0 ** 2))
        inner_term = np.maximum(inner_term, 1e-6)  # 避免根号内为负
        term2 = self.P1 * np.sqrt(inner_term)
        # 第三部分：0.5*d0*ρ0*s0*A0*||v_xy||³
        term3 = 0.5 * self.d0 * self.rho0 * self.s0 * self.A0 * (v_xy_norm ** 3)
        # 水平飞行总功率
        P_fly = term1 + term2 + term3

        # 3. 计算垂直飞行功率 P_ver (W)
        # P_ver = W*g*||v_z||
        P_ver = self.W * self.g * v_z_abs

        # 4. 计算总能耗 E = (P_fly + P_ver) * 飞行时间（焦耳）
        total_power = P_fly + P_ver
        energy = total_power * flight_time

        return energy

    def get_observation_and_state(self):
        flat_users = self.user_positions.flatten() / 300
        flat_uavs = self.uav_positions.flatten() / 300
        global_state = np.concatenate([flat_uavs, flat_users])

        observations = []

        for i in range(self.num_uavs):
            obs = np.concatenate([self.uav_positions[i], flat_users])
            # obs = np.concatenate([flat_uavs, flat_users])
            observations.append(obs)

        return np.array(observations), global_state

    def reset(self):
        self._set_random_seed()
        self.user_positions = self._generate_user_positions()
        self.uav_positions = self._generate_uav_position()
        self.user_uav_distances = self.get_user_all_uav_distances()
        np.random.seed(None)
        random.seed(None)
        return self.get_observation_and_state()

    def step(self, actions):
        # 1. 动作处理（适配三维速度）
        actions = np.clip(actions, -1, 1)  # 动作范围[-1,1]
        scale_factor = np.array([30, 30, 3])
        actions = actions * scale_factor

        # --- A. 位置更新与独立越界惩罚 ---
        # 计算预期新位置（三维）：当前位置 + 速度 * 飞行时间（1.5s）
        proposed_positions = self.uav_positions + actions * 1.5
        noise = np.random.normal(loc=0.0, scale=self.epsilon_0, size=proposed_positions.shape)
        proposed_positions += noise

        # 初始化边界惩罚 (每个 UAV 独有)
        boundary_penalties = np.zeros(self.num_uavs)

        # 检查是否越界：x/y∈[0, area_size], z∈[50,80]
        for i in range(self.num_uavs):
            x, y, z = proposed_positions[i]
            x_out = x < 0 or x > self.area_size[0]
            y_out = y < 0 or y > self.area_size[1]
            z_out = z < self.uav_height_min or z > self.uav_height_max
            # 任一维度越界则惩罚
            if x_out or y_out or z_out:
                boundary_penalties[i] = -4.0  # 越界惩罚系数

        # 限制位置在合法范围内（物理约束）
        proposed_positions[:, 0] = np.clip(proposed_positions[:, 0], 0.0, self.area_size[0])  # x
        proposed_positions[:, 1] = np.clip(proposed_positions[:, 1], 0.0, self.area_size[1])  # y
        proposed_positions[:, 2] = np.clip(proposed_positions[:, 2], self.uav_height_min, self.uav_height_max)  # z
        self.uav_positions = proposed_positions

        # --- B. 碰撞/过近距离独立惩罚（三维距离） ---
        # 计算UAV之间的三维距离
        uav_dists = cdist(self.uav_positions, self.uav_positions, metric='euclidean')
        np.fill_diagonal(uav_dists, np.inf)  # 忽略自身距离

        collision_penalties = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            min_dist = np.min(uav_dists[i])
            if min_dist < 3.0:  # 三维距离小于3米
                collision_penalties[i] = -5.0  # 碰撞惩罚系数

        # ==================== 新增 START：速度惩罚计算 ====================
        # 计算每个无人机的三维速度模（总速度）
        speed_magnitudes = np.linalg.norm(actions, axis=1)  # shape: (num_uavs,)，每个元素是对应无人机的总速度
        # 初始化速度惩罚（每个UAV独有）
        speed_penalties = np.zeros(self.num_uavs)
        speed_threshold = 20.0  # 速度阈值（m/s）
        speed_penalty_value = -5.0  # 速度超阈值的惩罚值（可根据需求调整）

        for i in range(self.num_uavs):
            if speed_magnitudes[i] > speed_threshold:
                speed_penalties[i] = speed_penalty_value


        # ==================== 2. 能耗惩罚计算（新增） ====================
        energy_penalties = np.zeros(self.num_uavs)
        flight_time = 1.5  # 对应公式中的T，与位置更新的飞行时间一致
        for i in range(self.num_uavs):
            # 计算第i个无人机的能耗
            uav_energy = self._calculate_uav_energy_consumption(actions[i], flight_time)
            # 能耗作为惩罚（能耗越大，惩罚越重，负号表示损失）
            energy_penalties[i] = -uav_energy * self.energy_penalty_coeff


        # 3. 生成三次新任务，使用GWOSolver计算总时延
        total_delay = 0
        # 新增：初始化每架无人机的处理时延（全程累加）
        uav_processing_latency = np.zeros(self.num_uavs)
        # 生成UAV失效状态（用于GWOSolver初始化）
        failure_status = self.simulate_uav_failure()
        # 更新用户-UAV连接（GWOSolver依赖该属性）
        self.update_user_uav_connections(failure_status)

        for _ in range(1):
            # 生成本次任务的size和complexity
            task_sizes = np.random.uniform(self.task_size_range[0], self.task_size_range[1], self.num_users)
            task_complexities = np.random.uniform(self.task_complexity_range[0], self.task_complexity_range[1],
                                                  self.num_users)
            # 赋值给env属性，供GWOSolver读取
            self.users_task_size = task_sizes
            self.users_task_complexity = task_complexities

            # 初始化GWOSolver求解当前任务
            solver = GWOSolver(self, self.args, failure_status)
            best_details, _ = solver.solve()

            # 累加本次任务的总时延
            if best_details is not None:
                # 解构best_details获取各参数
                lat_sum, solution, user_latencies, allocation = best_details
                # 判断所有用户时延是否都小于small_time_slot
                all_below_threshold = all(lat <= self.args.small_time_slot for lat in user_latencies)
                if all_below_threshold:
                    task_total_lat = lat_sum
                else:
                    # 超过阈值的用户时延设为10，其余保持不变，然后求和
                    adjusted_latencies = [3.0 if lat > self.args.small_time_slot else lat for lat in user_latencies]
                    task_total_lat = sum(adjusted_latencies)
                total_delay += task_total_lat

                # ==================== 新增核心逻辑：计算每架无人机的处理时延 ====================
                # 初始化本轮任务中每架无人机的临时处理时延
                uav_round_latency = np.zeros(self.num_uavs)
                # 遍历每个用户，将其时延分配到对应的无人机上
                for user_idx, uav_idx in enumerate(solution):
                    if uav_idx != -1:  # 排除未分配的用户
                        # 获取该用户的最终时延（与总时延计算逻辑一致）
                        user_lat = user_latencies[user_idx]
                        if user_lat > self.args.small_time_slot:
                            user_lat = 3.0  # 与总时延的调整逻辑对齐
                        # 累加到对应无人机的本轮时延
                        uav_round_latency[uav_idx] += user_lat
                # 将本轮无人机时延累加到全局无人机时延变量
                uav_processing_latency += uav_round_latency
                # =============================================================================
            else:
                # 若求解失败（如所有UAV失效），设置超大时延值
                total_delay += 1e2
                # 求解失败时，给所有存活无人机分配超大处理时延
                for uav_idx in range(self.num_uavs):
                    if failure_status[uav_idx] == 0:  # 仅存活无人机
                        uav_processing_latency[uav_idx] += 1e2

        # 全局时延奖励 (所有 UAV 共享)
        global_delay_reward = -total_delay
        delay_reward = -uav_processing_latency



        # 计算每个 UAV 的最终奖励
        final_rewards = np.zeros(self.num_uavs)
        final_costs = np.zeros(self.num_uavs)
        final_delays = np.zeros(self.num_uavs)
        final_energys = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            final_rewards[i] = delay_reward[i] + energy_penalties[i] + boundary_penalties[i] + speed_penalties[i] + collision_penalties[i]
            final_costs[i] = boundary_penalties[i] + speed_penalties[i] + collision_penalties[i]
            final_delays[i] = delay_reward[i]
            final_energys[i] = energy_penalties[i]

        obs, state = self.get_observation_and_state()
        done = False

        return obs, state, final_rewards,final_costs,final_delays,final_energys, done