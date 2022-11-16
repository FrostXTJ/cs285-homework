from collections import OrderedDict
import torch

from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure.sac_utils import soft_update_params
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # DONE: get this from previous HW  
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic  
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)

        next_action_distribution = self.actor(next_ob_no)
        next_action = next_action_distribution.rsample()
        log_probs = next_action_distribution.log_prob(next_action).sum(dim=1, keepdim=True)

        next_q1, next_q2 = self.critic_target(next_ob_no, next_action)
        next_q_min = torch.min(next_q1, next_q2)
        next_value = (next_q_min - self.actor.alpha * log_probs).squeeze(-1)
        
        q_target = re_n + self.gamma * (1 - terminal_n) * next_value
        q_target = q_target.unsqueeze(1)
        q_target = q_target.detach()

        q1, q2 = self.critic(ob_no, ac_na)
        critic_loss = self.critic.loss(q1, q_target) + self.critic.loss(q2, q_target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss.item()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # DONE: get this from previous HW
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging

        critic_loss = 0
        actor_loss = 0
        alpha_loss = 0
        alpha = 0

        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)

            if i % self.agent_params['critic_target_update_frequency'] == 0:
                with torch.no_grad():
                    soft_update_params(self.critic, self.critic_target, self.critic_tau)

            if i % self.agent_params['actor_update_frequency'] == 0:
                for j in range(self.agent_params['num_actor_updates_per_agent_update']):
                    actor_loss, alpha_loss, alpha = self.actor.update(ob_no, self.critic)

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        self.training_step += 1

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
