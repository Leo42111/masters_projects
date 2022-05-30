# -*- coding: utf-8 -*-
"""

@author: Leo Ma

Main function for A2C (baseline)

"""

import gym
import utils
import a2c
import pandas as pd
import torch


episodes = 2500
max_iter = 200  # Also set by env
ckpt_interval = 500
lr = 1e-4

# Initialize variables for recording 
G = []
total_losses = []
actor_losses = []
critic_losses = []

# Initialize the actor-critic network
agent = a2c.ActorCritic(3, 1, 256, lr=lr)

# Initialize the environment
env = gym.make("Pendulum")

# Loop the episodes
for eps in range(episodes):

    # Reset the environment and obtain first state
    current_state = env.reset()
    
    # Initialize records 
    rewards = []
    log_probs = []
    state_values = []

    # For each time step t
    for t in range(max_iter):
        env.render()
        
        # Get actions, probability, state value (& entropy) from agent
        action, action_log_prob, state_value = agent.get_action(current_state)
        
        # Save the obtained values for loss calculation later
        log_probs.append(action_log_prob)
        state_values.append(state_value)
        
        # Input the actions of agents &
        # get the rewards + if the game is finished
        current_state, reward, done, info = env.step(action)
        
        # Update reward records
        rewards.append(reward)
        
        # Check if it is the end of episode
        if done:
            print(f"Episode {eps+1} finished after {t+1} timesteps.")
            break
    
    # Get final Q value
    _, _, final_Q_value = agent.get_action(current_state)
    
    # Update the policy for the agent
    loss, actor_loss, critic_loss = agent.update_policy(rewards, log_probs, state_values, final_Q_value)
    
    # Calculate & save G for the agent in each episode
    G.append(a2c.calc_G_value(rewards))
    
    # Save the calculated combined loss
    total_losses.append(loss)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    
    # Save graphs & results in certain time interval
    if (eps+1) % ckpt_interval == 0:

        # Plot graphs for analysis
        filename_prefix = 'a2c'
        utils.plot_G_graph(filename_prefix, G)
        utils.plot_actor_loss_graph(filename_prefix, actor_losses)
        utils.plot_critic_loss_graph(filename_prefix, critic_losses)
        utils.plot_all_loss_graph(filename_prefix, total_losses, actor_losses, critic_losses)
               
        # Save results into a csv
        output_df = pd.DataFrame({'G': G,
                                  'total_loss': total_losses,
                                  'actor_loss': actor_losses,
                                  'critic_loss': critic_losses,
                                 })
        output_df.to_csv(f"./results/{filename_prefix}.csv")
        
        # Save models
        torch.save(agent.state_dict, f"./models/{filename_prefix}.model")
            
env.close()
