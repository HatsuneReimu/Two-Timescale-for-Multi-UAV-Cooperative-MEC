"""
修改：
1. 删除了原有的 GWOSolver 求解器。
2. 使用贪心算法替代 GWO，进行任务调度和计算资源分配。
3. [核心修改] 用户强制优先将任务上传给距离最近的存活无人机（接入无人机），
   然后贪心算法决定该任务是留在接入无人机处理，还是通过 A2A 链路中转给其他存活无人机处理。
4. 贪心策略尽量保证每架无人机服务的用户数 <= 5，并将存活无人机的最大算力全部分配给其下的用户。
"""

import numpy as np
from scipy.spatial.distance import cdist
import random


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
        self.users_task_size = np.zeros(self.num_users)
        self.users_task_complexity = np.zeros(self.num_users)

        # -------------------------- 3. 连接与任务分配相关参数 --------------------------
        self.user_uav_distances = np.full((self.num_users, self.num_uavs), np.inf)

        # -------------------------- 4. UAV相关参数 --------------------------
        self.uav_positions = np.zeros((self.num_uavs, 3))  # UAV三维位置 (num_uavs, 3)
        self.uav_tx_power = args.uav_tx_power
        self.uav_max_computing_capacity = args.uav_max_computing_capacity
        self.uav_height_min = args.min_height  # UAV最小高度
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
        self.epsilon_0 = 1  # 噪声的标准差

        # ==================== 能耗模型参数 ====================
        self.P0 = 80.0
        self.P1 = 100.0
        self.U_tips = 120.0
        self.v0 = 4.0
        self.d0 = 0.6
        self.rho0 = 1.225
        self.s0 = 0.08
        self.A0 = 0.5
        self.W = 2.0
        self.g = 9.81
        self.energy_penalty_coeff = 0.005

        self.random_seed = 123

    def _set_random_seed(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def simulate_uav_failure(self):
        failure_array = np.random.binomial(1, 0.01, self.num_uavs)
        if np.all(failure_array == 1):
            random_idx = np.random.randint(0, self.num_uavs)
            failure_array[random_idx] = 0
        return failure_array

    def _generate_user_positions(self):
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

        while len(uav_positions) < self.num_uavs:
            x = np.random.uniform(0, self.area_size[0])
            y = np.random.uniform(0, self.area_size[1])
            z = np.random.uniform(self.uav_height_min, self.uav_height_max)
            uav_positions.append([x, y, z])

        return np.array(uav_positions, dtype=np.float64)

    def get_user_all_uav_distances(self):
        user_pos_3d = np.hstack([self.user_positions, np.zeros((self.num_users, 1))])
        user_uav_distances = cdist(user_pos_3d, self.uav_positions, metric='euclidean').round(2)
        return user_uav_distances

    def calculate_los_probability(self):
        uav_2d_positions = self.uav_positions[:, :2]
        user_uav_level_distances = cdist(self.user_positions, uav_2d_positions, metric='euclidean').round(2)
        user_uav_level_distances = np.maximum(user_uav_level_distances, 1e-6)

        uav_heights = self.uav_positions[:, 2].reshape(1, -1)
        elevation_angle_rad = np.arctan(uav_heights / user_uav_level_distances)
        elevation_angle_deg = elevation_angle_rad * 180 / np.pi

        los_prob = 1 / (1 + 9.61 * np.exp(-0.168 * (elevation_angle_deg - 9.61)))
        return los_prob.round(4)

    def calculate_a2g_path_loss(self):
        los_prob = self.calculate_los_probability()
        uav_2d_positions = self.uav_positions[:, :2]
        user_uav_level_distances = cdist(self.user_positions, uav_2d_positions, metric='euclidean').round(2)
        uav_heights = self.uav_positions[:, 2].reshape(1, -1)

        straight_distance = np.sqrt(np.square(user_uav_level_distances) + np.square(uav_heights))
        straight_distance = np.maximum(straight_distance, 1e-6)

        fs_pl_term = 10 * np.log10(
            np.power((4 * np.pi * self.carrier_frequency_g2a * straight_distance) / self.light_speed,
                     self.path_loss_exponent))
        los_nlos_attenuation = los_prob * self.eta_los + (1 - los_prob) * self.eta_nlos
        total_pl = fs_pl_term + los_nlos_attenuation
        return total_pl.round(2)

    def calculate_a2a_path_loss(self):
        uav_uav_distances = cdist(self.uav_positions, self.uav_positions, metric='euclidean').round(2)
        uav_uav_distances = np.maximum(uav_uav_distances, 1e-6)

        fspi = 20 * np.log10((4 * np.pi * self.carrier_frequency_a2a * uav_uav_distances) / self.light_speed)
        np.fill_diagonal(fspi, 0.0)
        return fspi.round(2)

    def calculate_transmission_rate(self, pl_db, tx_power):
        pl_linear = np.power(10, pl_db / 10)
        pl_linear = np.maximum(pl_linear, 1e-6)

        noise_spectral_density_W_per_Hz = np.power(10, (self.noise_power - 30) / 10)
        sigma_square = noise_spectral_density_W_per_Hz * self.channel_bandwidth
        sigma_square = np.maximum(sigma_square, 1e-23)

        power_matrix = np.full_like(pl_db, tx_power, dtype=np.float64)
        if len(pl_db.shape) == 2 and pl_db.shape[0] == pl_db.shape[1]:
            np.fill_diagonal(power_matrix, 0.0)

        snr = power_matrix / (pl_linear * sigma_square)
        snr = np.minimum(snr, 1e6)
        transmission_rate = self.channel_bandwidth * np.log2(1 + snr)

        transmission_rate = np.maximum(transmission_rate, 0)
        return transmission_rate.round(2)

    def update_user_uav_connections(self, failure_status):
        self.user_uav_distances = self.get_user_all_uav_distances()
        dists = self.user_uav_distances.copy()
        failed = np.where(failure_status == 1)[0]
        if len(failed) > 0: dists[:, failed] = np.inf
        # 为每个用户找到距离最近的存活无人机作为接入网点
        self.user_uav_connections = np.argmin(dists, axis=1)
        self.user_uav_connections[np.min(dists, axis=1) == np.inf] = -1
        return self.user_uav_connections

    def _calculate_uav_energy_consumption(self, uav_velocity, flight_time):
        v_xy = uav_velocity[:2]
        v_z = uav_velocity[2]
        v_xy_norm = np.linalg.norm(v_xy)
        v_z_abs = np.abs(v_z)

        term1 = self.P0 * (1 + 3 * (v_xy_norm ** 2) / (self.U_tips ** 2))
        inner_term = 1 + (v_xy_norm ** 4) / (4 * (self.v0 ** 4)) - (v_xy_norm ** 2) / (2 * (self.v0 ** 2))
        inner_term = np.maximum(inner_term, 1e-6)
        term2 = self.P1 * np.sqrt(inner_term)
        term3 = 0.5 * self.d0 * self.rho0 * self.s0 * self.A0 * (v_xy_norm ** 3)
        P_fly = term1 + term2 + term3

        P_ver = self.W * self.g * v_z_abs
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

    # ==================== 基于上传与中转的贪心分配算法 ====================
    def _greedy_task_allocation(self, failure_status, task_sizes, task_complexities):
        active_uavs = np.where(failure_status == 0)[0]
        if len(active_uavs) == 0:
            return None, None

        # 预计算通信速率
        pl_g2a = self.calculate_a2g_path_loss()
        rate_g2a = self.calculate_transmission_rate(pl_g2a, self.user_tx_power)
        pl_a2a = self.calculate_a2a_path_loss()
        rate_a2a = self.calculate_transmission_rate(pl_a2a, self.uav_tx_power)

        # 任务计算量 W = size * complexity
        W_vec = task_sizes * task_complexities
        sqrt_W_vec = np.sqrt(W_vec)

        # 记录每架存活UAV分配到的用户列表 (表示由该UAV进行计算处理)
        uav_users = {u: [] for u in active_uavs}
        solution = np.full(self.num_users, -1)

        # 贪心策略：优先分配计算量大的任务（倒序排序）
        sorted_users = np.argsort(W_vec)[::-1]

        for m in sorted_users:
            # 1. 用户强制接入最近的存活无人机
            acc_uav = self.user_uav_connections[m]
            if acc_uav == -1:
                continue  # 无可用接入，直接跳过，时延后续补极大值

            # 计算基础的 G2A 上传时延
            r_up = rate_g2a[m, acc_uav]
            t_up = task_sizes[m] / r_up if r_up > 1e-5 else 1e6

            best_uav = -1
            best_score = float('inf')

            # 检查是否还有UAV尚未满载（<5个用户）
            has_capacity = any(len(uav_users[cand]) < 5 for cand in active_uavs)

            for u in active_uavs:
                # 尽量不超过5个用户；如果全都超过了，则放宽限制
                if has_capacity and len(uav_users[u]) >= 5:
                    continue

                # 2. 估算中转传输时延 (A2A)
                t_trans = 0
                if acc_uav != u:  # 如果处理节点不是接入节点，产生中转时延
                    r_tr = rate_a2a[acc_uav, u]
                    if r_tr > 1e-5:
                        t_trans = task_sizes[m] / r_tr
                    else:
                        t_trans = 1e6

                # 3. 估算计算时延（假设分配后，该UAV计算资源按sqrt(W)完全分配）
                existing_sqrt_sum = sum(sqrt_W_vec[user] for user in uav_users[u])
                new_sum_sqrt = existing_sqrt_sum + sqrt_W_vec[m]
                est_f = (sqrt_W_vec[m] / new_sum_sqrt) * self.uav_max_computing_capacity
                t_comp_est = W_vec[m] / est_f if est_f > 0 else 1e6

                # 贪心分数：上传时延 + 中转时延 + 预估计算时延
                score = t_up + t_trans + t_comp_est

                if score < best_score:
                    best_score = score
                    best_uav = u

            # 将任务分配到总时延最小的处理无人机
            if best_uav != -1:
                solution[m] = best_uav
                uav_users[best_uav].append(m)

        # ================== 计算真实闭环时延并分配全部计算资源 ==================
        user_latencies = np.full(self.num_users, 1e6)  # 默认初始化为极大值
        for u in active_uavs:
            users = uav_users[u]
            if not users:
                continue

            sum_sqrt = sum(sqrt_W_vec[m] for m in users)
            for m in users:
                acc_uav = self.user_uav_connections[m]

                # 1. 接入上传时延 (G2A)
                r_up = rate_g2a[m, acc_uav]
                t_up = task_sizes[m] / r_up if r_up > 1e-5 else 1e6

                # 2. 中转传输时延 (A2A)
                t_trans = 0
                if acc_uav != u:
                    r_tr = rate_a2a[acc_uav, u]
                    t_trans = task_sizes[m] / r_tr if r_tr > 1e-5 else 1e6

                # 3. 计算时延 (完全分配该无人机的算力)
                f_m = (sqrt_W_vec[m] / sum_sqrt) * self.uav_max_computing_capacity
                t_comp = W_vec[m] / f_m if f_m > 0 else 1e6

                # 真实总时延
                user_latencies[m] = t_up + t_trans + t_comp

        return solution, user_latencies

    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        scale_factor = np.array([30, 30, 3])
        actions = actions * scale_factor

        # --- A. 位置更新与独立越界惩罚 ---
        proposed_positions = self.uav_positions + actions * 1.5
        noise = np.random.normal(loc=0.0, scale=self.epsilon_0, size=proposed_positions.shape)
        proposed_positions += noise

        boundary_penalties = np.zeros(self.num_uavs)

        for i in range(self.num_uavs):
            x, y, z = proposed_positions[i]
            x_out = x < 0 or x > self.area_size[0]
            y_out = y < 0 or y > self.area_size[1]
            z_out = z < self.uav_height_min or z > self.uav_height_max
            if x_out or y_out or z_out:
                boundary_penalties[i] = -4.0

        proposed_positions[:, 0] = np.clip(proposed_positions[:, 0], 0.0, self.area_size[0])
        proposed_positions[:, 1] = np.clip(proposed_positions[:, 1], 0.0, self.area_size[1])
        proposed_positions[:, 2] = np.clip(proposed_positions[:, 2], self.uav_height_min, self.uav_height_max)
        self.uav_positions = proposed_positions

        # --- B. 碰撞/过近距离独立惩罚（三维距离） ---
        uav_dists = cdist(self.uav_positions, self.uav_positions, metric='euclidean')
        np.fill_diagonal(uav_dists, np.inf)

        collision_penalties = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            min_dist = np.min(uav_dists[i])
            if min_dist < 3.0:
                collision_penalties[i] = -5.0

                # --- C. 速度惩罚计算 ---
        speed_magnitudes = np.linalg.norm(actions, axis=1)
        speed_penalties = np.zeros(self.num_uavs)
        speed_threshold = 20.0
        speed_penalty_value = -5.0

        for i in range(self.num_uavs):
            if speed_magnitudes[i] > speed_threshold:
                speed_penalties[i] = speed_penalty_value

        # --- D. 能耗惩罚计算 ---
        energy_penalties = np.zeros(self.num_uavs)
        flight_time = 1.5
        for i in range(self.num_uavs):
            uav_energy = self._calculate_uav_energy_consumption(actions[i], flight_time)
            energy_penalties[i] = -uav_energy * self.energy_penalty_coeff

        # ==================== 生成新任务并使用贪心算法调度 ====================
        total_delay = 0
        uav_processing_latency = np.zeros(self.num_uavs)

        failure_status = self.simulate_uav_failure()
        self.update_user_uav_connections(failure_status)

        for _ in range(1):
            task_sizes = np.random.uniform(self.task_size_range[0], self.task_size_range[1], self.num_users)
            task_complexities = np.random.uniform(self.task_complexity_range[0], self.task_complexity_range[1],
                                                  self.num_users)

            self.users_task_size = task_sizes
            self.users_task_complexity = task_complexities

            # --- 调用贪心算法进行任务调度 ---
            solution, user_latencies = self._greedy_task_allocation(failure_status, task_sizes, task_complexities)

            if solution is not None:
                lat_sum = np.sum(user_latencies)

                # 若全员满足阈值，使用总时延；否则进行超量惩罚裁减
                all_below_threshold = all(lat <= self.args.small_time_slot for lat in user_latencies)
                if all_below_threshold:
                    task_total_lat = lat_sum
                else:
                    adjusted_latencies = [3.0 if lat > self.args.small_time_slot else lat for lat in user_latencies]
                    task_total_lat = sum(adjusted_latencies)

                total_delay += task_total_lat

                # 统计计算节点(UAV)承担的时延压力
                uav_round_latency = np.zeros(self.num_uavs)
                for user_idx, uav_idx in enumerate(solution):
                    if uav_idx != -1:
                        user_lat = user_latencies[user_idx]
                        if user_lat > self.args.small_time_slot:
                            user_lat = 3.0
                        uav_round_latency[uav_idx] += user_lat

                uav_processing_latency += uav_round_latency
            else:
                # 极端情况：无存活UAV
                total_delay += 1e2
                for uav_idx in range(self.num_uavs):
                    if failure_status[uav_idx] == 0:
                        uav_processing_latency[uav_idx] += 1e2

        global_delay_reward = -total_delay
        delay_reward = -uav_processing_latency

        final_rewards = np.zeros(self.num_uavs)
        final_costs = np.zeros(self.num_uavs)
        final_delays = np.zeros(self.num_uavs)
        final_energys = np.zeros(self.num_uavs)

        for i in range(self.num_uavs):
            final_rewards[i] = delay_reward[i] + energy_penalties[i] + boundary_penalties[i] + speed_penalties[i] + \
                               collision_penalties[i]
            final_costs[i] = boundary_penalties[i] + speed_penalties[i] + collision_penalties[i]
            final_delays[i] = delay_reward[i]
            final_energys[i] = energy_penalties[i]

        obs, state = self.get_observation_and_state()
        done = False


        return obs, state, final_rewards, final_costs, final_delays, final_energys, done