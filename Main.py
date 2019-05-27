import csv
import numpy as np
from task import Task 
from DDGP import DDPG 
import sys
import matplotlib.pyplot as plt

# Modify the values below to give the quadcopter a different starting position.
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 0., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results
target_pose = np.array([0., 0., 10.])
num_episodes = 2
Best_episode = 0
Best_score = -10000.0
# Setup
task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
agent = DDPG(task)
done = False
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {} 

reward_labels = ['episode', 'reward']
reward_results = {x : [] for x in reward_labels}
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        results[i_episode] = {x : [] for x in labels}
        score = 0
        while True:
            action = agent.act(state) 
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action)

            for ii in range(len(labels)):
                results[i_episode][labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            state = next_state
            score += reward
            if done:
                if score > Best_score:
                    Best_score = score
                    Best_episode = i_episode
                print("\rEpisode = {:4d}, score = {:7.3f}, Best_score = {:7.3f}, Best_episode = {:4d}".format(
                i_episode, score, Best_score, Best_episode), end="")
                break
        reward_results['episode'].append(i_episode)
        reward_results['reward'].append(score)
        reward_results['episode'][0]
        sys.stdout.flush()

    plt.plot(results[10]['time'], results[10]['x'], label='x')
    plt.plot(results[10]['time'], results[10]['y'], label='y')
    plt.plot(results[10]['time'], results[10]['z'], label='z')
    plt.legend()
    _ = plt.ylim()




