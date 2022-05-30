# -*- coding: utf-8 -*-
"""

@author: Leo Ma

A separate python file for common functions

"""


# Import libraries and define global constants
from matplotlib import pyplot as plt

# Define common graph plotting parameters
fig_size = (10,6)
x_label = "Episode"
plt.rcParams.update({'font.size': 16})

# Function to plot total rewards over episodes
def plot_G_graph(filename_prefix, G_list):
    plt.figure(figsize=fig_size)
    plt.plot(G_list)
    plt.title(f"Total discounted rewards for agent over {len(G_list)} episodes")
    plt.xlabel(x_label)
    plt.ylabel("Total return (G)")
    filename = f"./graphs/{filename_prefix}_G.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    

# Function to plot the actor loss over episodes
def plot_actor_loss_graph(filename_prefix, actor_losses):
    plt.figure(figsize=fig_size)
    plt.plot(actor_losses, color='red', label='Actor')
    plt.title(f"Actor loss for agent over {len(actor_losses)} episodes")
    plt.xlabel(x_label)
    plt.ylabel("Loss")
    filename = f"./graphs/{filename_prefix}_actor_loss.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


# Function to plot the critic loss over episodes
def plot_critic_loss_graph(filename_prefix, critic_losses):
    plt.figure(figsize=fig_size)
    plt.plot(critic_losses, color='blue', label='critic')
    plt.title(f"Actor loss for agent over {len(critic_losses)} episodes")
    plt.xlabel(x_label)
    plt.ylabel("Loss")
    filename = f"./graphs/{filename_prefix}_critic_loss.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    

# Function to plot all the losses (total, actor, critic) over episodes
def plot_all_loss_graph(filename_prefix, total_losses, actor_losses, critic_losses):
    plt.figure(figsize=fig_size)
    plt.plot(total_losses, color='black', label='Total')
    plt.plot(actor_losses, color='red', label='Actor')
    plt.plot(critic_losses, color='blue', label='critic')
    plt.axhline(y = 0, color = 'grey', linestyle = '--', alpha=0.6)
    plt.title(f"Loss for agent over {len(total_losses)} episodes")
    plt.xlabel(x_label)
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    filename = f"./graphs/{filename_prefix}_all_loss.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

# Function to plot total rewards of all the MC A2C models
def plot_four_G_graph(baseline, separate, reg, mae):
    plt.figure(figsize=(10,8))
    plt.plot(baseline, color='black', label='Baseline', alpha=0.5)
    plt.plot(separate, color='red', label='Separate', alpha=0.5)
    plt.plot(reg, color='green', label='Reg', alpha=0.5)
    plt.plot(mae, color='blue', label='Reg + MAE', alpha=0.5)
    plt.title(f"Total rewards for agent over {len(baseline)} episodes")
    plt.xlabel('Episodes')
    plt.ylabel("Total return (G)")
    plt.legend(loc="upper right")
    filename = "./graphs/models_G.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

# Function to plot total loss of all the MC A2C models
def plot_four_loss_graph(baseline, separate, reg, mae):
    plt.figure(figsize=(10,8))
    plt.plot(baseline, color='black', label='Baseline', alpha=0.5)
    plt.plot(separate, color='red', label='Separate', alpha=0.5)
    plt.plot(reg, color='green', label='Reg', alpha=0.5)
    plt.plot(mae, color='blue', label='Reg + MAE', alpha=0.5)
    plt.title(f"Total loss for agent over {len(baseline)} episodes")
    plt.xlabel('Episodes')
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    filename = "./graphs/models_loss.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
