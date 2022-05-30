# -*- coding: utf-8 -*-
"""

@author: Leo Ma

Implementation on A2C (Advantage Actor Critic)

"""


# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim


# Initialize hyperparameters
gamma = 0.99  # discount factor


# Define the policy network class for A2C
class ActorCritic(nn.Module):
    
    # Initialize ActorCritic class
    def __init__(self, input_dim, num_actions, hidden_dim, lr):
        super(ActorCritic, self).__init__()
        
        # Same linear layer for both the actor and critic
        self.init_ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            )
        
        # Different linear layer for actor and critic
        # Actor - output nodes = 1 (mean of normal distribution)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            )
        
        # Critic - output nodes = 1 (state value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Separate parameter for the standard deviation of normal distribution
        actor_std = nn.Parameter(torch.full((num_actions,), 0.1))
        self.register_parameter("actor_std", actor_std)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    # Function to output:
    # - State value V from critic
    # - normal probability distribution of actions from actor
    def forward(self, state):
        out = self.init_ff(state)
        state_value = self.critic(out)
        
        action_mean = self.actor(out)
        action_std = torch.clamp(self.actor_std.exp(), 1e-3, 10)
        action_dist = torch.distributions.Normal(action_mean, action_std)
                
        return state_value, action_dist

    # Function to get the action from policy network
    def get_action(self, state):
        
        # Turn the input state into tensor
        state_tensor = torch.FloatTensor(state).view(-1)
        
        # Pass the state through the network to get action_dist & state_value
        state_value, action_dist = self.forward(state_tensor)
        
        # Sample an action from the probability distribution
        action = torch.clamp(action_dist.sample().detach(), min=-2.0, max=2.0)
        action_log_prob = action_dist.log_prob(action)
 
        # Output: 
        # action(float) - sampled action
        # log_prob(tensor) - the log probability of the selected action from the actor
        # state_value(tensor) - the state value predicted by the critic
        return [action.item()], action_log_prob, state_value
    
    # Function to update the policy
    def update_policy(self, rewards_list, log_probs, state_values, final_Q_value):
        
        # Calculate Q-values for each time step t
        Q_values = []
        Q_value = final_Q_value
        for t in reversed(range(len(rewards_list))):
            Q_value = rewards_list[t] + gamma * Q_value
            Q_values.insert(0, Q_value)
        
        # Turn all lists into tensors for preparing the calculation
        Q_values = torch.tensor(Q_values)
        V_values = torch.cat(state_values)
        log_probs = torch.cat(log_probs)
        
        # Calculate the advantage
        advantage = Q_values - V_values
        
        # Calculate the loss
        # Papers mentioned accumulate gradient, i.e. sum()
        # Some codes used mean() instead
        # We use mean() in this project
        policy_losses = (-log_probs * advantage.detach()).mean()
        value_losses = advantage.pow(2).mean()
        loss = policy_losses + value_losses
        
        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return the policy gradient for analysis
        return loss.item(), policy_losses.item(), value_losses.item()


# Fucntion to calculate total discounted reward G
def calc_G_value(rewards_list):
    current_discount = 1.0
    G_value = 0.0
    for reward in rewards_list:
        G_value += current_discount * reward
        current_discount *= gamma
    return G_value

