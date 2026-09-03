import numpy as np
import random
import math  # 新增，用于计算莱维飞行(Levy Flight)的伽马函数


class DHHOSolver:  # 保持类名不变，以兼容 ENV 类的调用，实质为 DHHO 求解器
    def __init__(self, env, args, failure_status):
        self.env = env
        self.args = args
        self.failure_status = failure_status
        self.active_uavs = np.where(failure_status == 0)[0]
        self.T_SMALL = args.small_time_slot

        # 预计算速率 (Rate Matrices)
        pl_g2a = env.calculate_a2g_path_loss()
        self.rate_g2a = env.calculate_transmission_rate(pl_g2a, env.user_tx_power)
        pl_a2a = env.calculate_a2a_path_loss()
        self.rate_a2a = env.calculate_transmission_rate(pl_a2a, env.uav_tx_power)

        # 预计算任务的计算量 W 和 sqrt(W)
        self.W_vec = self.env.users_task_size * self.env.users_task_complexity
        self.sqrt_W_vec = np.sqrt(self.W_vec)

    def _get_greedy_solution(self):
        """贪心初始化策略"""
        if len(self.active_uavs) == 0:
            return np.full(self.env.num_users, -1)

        uav_users = {u: [] for u in self.active_uavs}
        solution = np.full(self.env.num_users, -1)

        # 优先处理计算量大的任务
        sorted_users = np.argsort(self.W_vec)[::-1]

        for m in sorted_users:
            acc_uav = self.env.user_uav_connections[m]
            if acc_uav == -1:
                continue

            r_up = self.rate_g2a[m, acc_uav]
            t_up = self.env.users_task_size[m] / r_up if r_up > 1e-5 else 1e6

            best_uav = -1
            best_score = float('inf')

            # 软性负载均衡：尽量不要让单架UAV负载过多用户
            has_capacity = any(len(uav_users[cand]) < 5 for cand in self.active_uavs)

            for u in self.active_uavs:
                if has_capacity and len(uav_users[u]) >= 5:
                    continue

                # 中转时延
                t_trans = 0
                if acc_uav != u:
                    r_tr = self.rate_a2a[acc_uav, u]
                    t_trans = self.env.users_task_size[m] / r_tr if r_tr > 1e-5 else 1e6

                # 估算计算时延 (假设加入该UAV后，利用闭式解分配)
                existing_sqrt_sum = sum(self.sqrt_W_vec[user] for user in uav_users[u])
                new_sum_sqrt = existing_sqrt_sum + self.sqrt_W_vec[m]
                est_f = (self.sqrt_W_vec[m] / new_sum_sqrt) * self.env.uav_max_computing_capacity
                t_comp_est = self.W_vec[m] / est_f if est_f > 0 else 1e6

                score = t_up + t_trans + t_comp_est
                if score < best_score:
                    best_score = score
                    best_uav = u

            if best_uav != -1:
                solution[m] = best_uav
                uav_users[best_uav].append(m)

        return solution

    def heuristic_repair(self, uav_indices):
        """两阶段启发式局部搜索与修复"""
        if len(self.active_uavs) <= 1:
            return uav_indices

        for _ in range(2):
            uav_load_sum_sqrt = np.zeros(self.env.num_uavs)
            for m, u_idx in enumerate(uav_indices):
                if u_idx != -1:
                    uav_load_sum_sqrt[u_idx] += self.sqrt_W_vec[m]

            current_latencies = np.zeros(self.env.num_users)
            violators = []

            # 1. 评估当前状态
            for m in range(self.env.num_users):
                u = uav_indices[m]
                if u == -1:
                    violators.append(m)
                    continue
                lat = self._calculate_single_user_latency(m, u, uav_load_sum_sqrt[u])
                current_latencies[m] = lat
                if lat > self.T_SMALL:
                    violators.append(m)

            # 阶段1：尝试修复超时的用户
            for m in violators:
                best_uav = uav_indices[m]
                min_lat = current_latencies[m] if best_uav != -1 else float('inf')
                found_feasible = False if min_lat > self.T_SMALL else True

                for u_cand in self.active_uavs:
                    if u_cand == uav_indices[m]:
                        continue
                    new_load_cand = uav_load_sum_sqrt[u_cand] + self.sqrt_W_vec[m]
                    est_lat = self._calculate_single_user_latency(m, u_cand, new_load_cand)

                    if est_lat <= self.T_SMALL:
                        if not found_feasible or est_lat < min_lat:
                            best_uav = u_cand
                            min_lat = est_lat
                            found_feasible = True
                    elif not found_feasible and est_lat < min_lat:
                        best_uav = u_cand
                        min_lat = est_lat

                if best_uav != -1 and best_uav != uav_indices[m]:
                    old_u = uav_indices[m]
                    if old_u != -1: uav_load_sum_sqrt[old_u] -= self.sqrt_W_vec[m]
                    uav_indices[m] = best_uav
                    uav_load_sum_sqrt[best_uav] += self.sqrt_W_vec[m]

            # 阶段2：贪婪优化
            if not violators:
                worst_user = np.argmax(current_latencies)
                worst_lat = current_latencies[worst_user]
                best_uav = uav_indices[worst_user]

                for u_cand in self.active_uavs:
                    if u_cand == best_uav: continue
                    new_load_cand = uav_load_sum_sqrt[u_cand] + self.sqrt_W_vec[worst_user]
                    est_lat = self._calculate_single_user_latency(worst_user, u_cand, new_load_cand)
                    if est_lat < worst_lat and est_lat <= self.T_SMALL:
                        best_uav = u_cand
                        worst_lat = est_lat

                if best_uav != uav_indices[worst_user]:
                    uav_indices[worst_user] = best_uav
                break

        return uav_indices

    def _calculate_single_user_latency(self, m, u, uav_sum_sqrt_W):
        """辅助函数：计算单个用户 m 在 UAV u 上特定负载下的总时延"""
        acc_uav = self.env.user_uav_connections[m]
        if acc_uav == -1: return 1e6

        r_up = self.rate_g2a[m, acc_uav]
        if r_up < 1e-5: return 1e6
        t_up = self.env.users_task_size[m] / r_up

        t_trans = 0
        if acc_uav != u:
            r_tr = self.rate_a2a[acc_uav, u]
            if r_tr < 1e-5: return 1e6
            t_trans = self.env.users_task_size[m] / r_tr

        if uav_sum_sqrt_W <= 0: return 1e6
        t_comp = (self.sqrt_W_vec[m] * uav_sum_sqrt_W) / self.env.uav_max_computing_capacity

        return t_up + t_trans + t_comp

    def calculate_objective(self, positions_continuous):
        """计算适应度并返回详细信息"""
        uav_indices = np.round(positions_continuous).astype(int)
        uav_indices = self._correct_positions(uav_indices)

        uav_indices = self.heuristic_repair(uav_indices)

        total_lat = 0
        penalty = 0
        user_lats = np.zeros(self.env.num_users)
        allocation_scheme = {}

        uav_tasks = {u: [] for u in range(self.env.num_uavs)}
        for m, u in enumerate(uav_indices):
            if u != -1: uav_tasks[u].append(m)

        for u in range(self.env.num_uavs):
            users = uav_tasks[u]
            if not users:
                allocation_scheme[u] = []
                continue

            sum_sqrt = sum(self.sqrt_W_vec[m] for m in users)

            f_allocs = []
            if sum_sqrt > 0:
                for m in users:
                    f = (self.sqrt_W_vec[m] / sum_sqrt) * self.env.uav_max_computing_capacity
                    f_allocs.append(f)
            else:
                f_allocs = [0] * len(users)

            allocation_scheme[u] = f_allocs

            for i, m in enumerate(users):
                lat = self._calculate_single_user_latency(m, u, sum_sqrt)
                user_lats[m] = lat

                if lat > self.T_SMALL:
                    penalty += 5000 * (lat - self.T_SMALL) + 1000

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

    def _get_levy(self, dim):
        """莱维飞行 (Levy Flight) 步长生成"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / np.power(np.abs(v), 1 / beta)
        return step

    def solve(self, pop_size=30, max_iter=100):
        """核心求解器：已被升级为离散哈里斯鹰优化(DHHO)"""
        if len(self.active_uavs) == 0:
            return None, []

        dim = self.env.num_users
        positions = np.zeros((pop_size, dim))
        fitness = np.full(pop_size, float("inf"))

        # [贪心种子注入]
        greedy_seed = self._get_greedy_solution()

        for i in range(pop_size):
            if i == 0:
                positions[i, :] = greedy_seed
            elif i < pop_size // 4:
                positions[i, :] = greedy_seed + np.random.uniform(-0.5, 0.5, dim)
            else:
                for d in range(dim):
                    positions[i, d] = random.choice(self.active_uavs) + np.random.uniform(-0.49, 0.49)

        # 猎物(兔子)的最佳位置，即全局最优
        rabbit_pos = np.zeros(dim)
        rabbit_score = float("inf")
        best_details = None
        convergence_curve = []

        for t in range(max_iter):
            # 1. 评估种群并寻找猎物(最优解)
            for i in range(pop_size):
                positions[i, :] = np.clip(positions[i, :], 0, self.env.num_uavs - 1)

                fit, true_lat, discrete_sol, u_lats, scheme = self.calculate_objective(positions[i, :])

                # 核心机制：将经过贪婪微调后的有效整数解强制写回连续空间，避免算法假收敛
                positions[i, :] = discrete_sol.astype(float)
                fitness[i] = fit

                if fit < rabbit_score:
                    rabbit_score = fit
                    rabbit_pos = positions[i, :].copy()
                    best_details = (true_lat, discrete_sol, u_lats, scheme)

            convergence_curve.append(rabbit_score)

            # 2. HHO 种群状态更新
            X_mean = np.mean(positions, axis=0)

            for i in range(pop_size):
                # 猎物能量 E (递减公式: E = 2*E0*(1 - t/T))
                E0 = 2 * random.random() - 1  # [-1, 1]
                E = 2 * E0 * (1 - t / max_iter)
                abs_E = abs(E)

                if abs_E >= 1:
                    # --- 探索阶段 (Exploration) ---
                    q = random.random()
                    rand_idx = random.randint(0, pop_size - 1)
                    X_rand = positions[rand_idx, :].copy()

                    if q >= 0.5:
                        # 策略 1: 根据其他哈里斯鹰位置进行探索
                        r1, r2 = random.random(), random.random()
                        positions[i, :] = X_rand - r1 * np.abs(X_rand - 2 * r2 * positions[i, :])
                    else:
                        # 策略 2: 在当前猎物与平均位置附近探索
                        r3, r4 = random.random(), random.random()
                        LB, UB = 0, self.env.num_uavs - 1
                        positions[i, :] = (rabbit_pos - X_mean) - r3 * (LB + r4 * (UB - LB))

                else:
                    # --- 开发阶段 (Exploitation) ---
                    r = random.random()  # 猎物逃脱概率
                    J = 2 * (1 - random.random())  # 猎物随机跳跃幅度

                    if r >= 0.5 and abs_E >= 0.5:
                        # 软包围 (Soft besiege)
                        positions[i, :] = rabbit_pos - E * np.abs(J * rabbit_pos - positions[i, :])

                    elif r >= 0.5 and abs_E < 0.5:
                        # 硬包围 (Hard besiege)
                        positions[i, :] = rabbit_pos - E * np.abs(rabbit_pos - positions[i, :])

                    elif r < 0.5 and abs_E >= 0.5:
                        # 渐进式快速俯冲软包围 (Soft besiege with progressive rapid dives)
                        Y = rabbit_pos - E * np.abs(J * rabbit_pos - positions[i, :])
                        Y_clipped = np.clip(Y, 0, self.env.num_uavs - 1)
                        fit_Y, true_lat_Y, sol_Y, u_lats_Y, scheme_Y = self.calculate_objective(Y_clipped)

                        if fit_Y < fitness[i]:
                            positions[i, :] = sol_Y.astype(float)
                            fitness[i] = fit_Y
                            # 即时更新猎物
                            if fit_Y < rabbit_score:
                                rabbit_score, rabbit_pos = fit_Y, sol_Y.astype(float)
                                best_details = (true_lat_Y, sol_Y, u_lats_Y, scheme_Y)
                        else:
                            # 莱维飞行 (Levy Flight) 突袭
                            Z = Y + np.random.randn(dim) * self._get_levy(dim)
                            Z_clipped = np.clip(Z, 0, self.env.num_uavs - 1)
                            fit_Z, true_lat_Z, sol_Z, u_lats_Z, scheme_Z = self.calculate_objective(Z_clipped)
                            if fit_Z < fitness[i]:
                                positions[i, :] = sol_Z.astype(float)
                                fitness[i] = fit_Z
                                if fit_Z < rabbit_score:
                                    rabbit_score, rabbit_pos = fit_Z, sol_Z.astype(float)
                                    best_details = (true_lat_Z, sol_Z, u_lats_Z, scheme_Z)

                    elif r < 0.5 and abs_E < 0.5:
                        # 渐进式快速俯冲硬包围 (Hard besiege with progressive rapid dives)
                        Y = rabbit_pos - E * np.abs(J * rabbit_pos - X_mean)
                        Y_clipped = np.clip(Y, 0, self.env.num_uavs - 1)
                        fit_Y, true_lat_Y, sol_Y, u_lats_Y, scheme_Y = self.calculate_objective(Y_clipped)

                        if fit_Y < fitness[i]:
                            positions[i, :] = sol_Y.astype(float)
                            fitness[i] = fit_Y
                            if fit_Y < rabbit_score:
                                rabbit_score, rabbit_pos = fit_Y, sol_Y.astype(float)
                                best_details = (true_lat_Y, sol_Y, u_lats_Y, scheme_Y)
                        else:
                            Z = Y + np.random.randn(dim) * self._get_levy(dim)
                            Z_clipped = np.clip(Z, 0, self.env.num_uavs - 1)
                            fit_Z, true_lat_Z, sol_Z, u_lats_Z, scheme_Z = self.calculate_objective(Z_clipped)
                            if fit_Z < fitness[i]:
                                positions[i, :] = sol_Z.astype(float)
                                fitness[i] = fit_Z
                                if fit_Z < rabbit_score:
                                    rabbit_score, rabbit_pos = fit_Z, sol_Z.astype(float)
                                    best_details = (true_lat_Z, sol_Z, u_lats_Z, scheme_Z)

        return best_details, convergence_curve