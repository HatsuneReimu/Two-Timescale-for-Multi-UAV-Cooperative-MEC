import torch
from CAMRL.algorithms.actor_critic import R_Actor, R_Critic
from CAMRL.algorithms.utils.util import update_linear_schedule

class MAPPOPolicy_L:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.cost_critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(),
                                               lr=self.critic_lr,
                                               eps=self.opti_eps,
                                               weight_decay=self.weight_decay)


    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.cost_optimizer, episode, episodes, self.critic_lr)


    def get_actions(self, cent_obs, obs,available_actions=None,deterministic=False):
        actions, action_log_probs = self.actor(obs,
                                                available_actions,
                                                deterministic)

        values = self.critic(cent_obs)
        cost_values = self.cost_critic(cent_obs)


        return values, cost_values, actions, action_log_probs


    def get_values(self, cent_obs):
        values = self.critic(cent_obs)
        return values


    def get_cost_values(self, cent_obs):
        cost_values = self.cost_critic(cent_obs)
        return cost_values


    def evaluate_actions(self, cent_obs, obs, action, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     action,
                                                                     available_actions,
                                                                     active_masks)

        values = self.critic(cent_obs)
        cost_values = self.cost_critic(cent_obs)

        return values, cost_values, action_log_probs, dist_entropy


    def act(self, obs, available_actions=None, deterministic=False):
        actions, _ = self.actor(obs, available_actions, deterministic)
        return actions
