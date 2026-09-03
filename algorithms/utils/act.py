import torch
import torch.nn as nn
import sys

try:
    from CAMRL.algorithms.utils.distributions import DiagGaussian
except Exception as e:
    print(f"❌ 导入过程中出现未知错误：{e}")
    sys.exit(1)

class ACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()

        action_dim = action_space
        self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)
    
    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        dist_entropy = action_logits.entropy().mean()
        
        return action_log_probs, dist_entropy


    def evaluate_actions_trpo(self, x, action, available_actions=None, active_masks=None):
        action_logits = self.action_out(x, available_actions)
        action_mu = action_logits.mean
        action_std = action_logits.stddev
        action_log_probs = action_logits.log_probs(action)
        dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy, action_mu, action_std


