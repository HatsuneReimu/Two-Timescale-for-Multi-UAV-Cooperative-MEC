import torch
import torch.nn as nn
import numpy as np
import sys

try:
    from CAMRL.algorithms.utils.mlp import MLPBase
    from CAMRL.algorithms.utils.act import ACTLayer
except Exception as e:
    # 捕获其他意外异常（兜底，可选）
    print(f"❌ 导入过程中出现未知错误：{e}")
    sys.exit(1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.base = MLPBase(args, obs_space)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        self.to(device)

    def forward(self, obs, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs

    def evaluate_actions(self, obs, action, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)


        if self.args.algorithm_name == "macpo":
            action_log_probs, dist_entropy, action_mu, action_std = self.act.evaluate_actions_trpo(actor_features,
                                                                       action, available_actions,
                                                                       active_masks=
                                                                       active_masks if self._use_policy_active_masks
                                                                       else None)

            return action_log_probs, dist_entropy, action_mu, action_std
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                       action, available_actions,
                                                                       active_masks=
                                                                       active_masks if self._use_policy_active_masks
                                                                       else None)
            return action_log_probs, dist_entropy



class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        self.base = MLPBase(args, cent_obs_space)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values