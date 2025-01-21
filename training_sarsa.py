import numpy as np
import gymnasium as gym
import pickle
import time
import random

from Minigrid import minigrid
from training import representation, preprocess_observation, serialize_state, pourcent_action_count, epsilon_greedy_policy, plot_courbes



def sarsa_1_seed(env, Q, episodes, alpha, gamma, epsilon, max_steps, training_seed, amplification, save_data=False, plot=False):
    """Execute une sequence d'entrainement pour une seed complete avec SARSA.

    plot_courbes (bool, optional): Est-ce qu'on veut afficher les courbes ou pas. Defaults to False.
    """
    action_space = env.action_space

    momentum = 10  # Nombre de fois qu'on va faire encore 100 episodes lorsqu'on atteint un moment ou le training est "debloque"
    t1 = time.time()
    batch_reward = [0 for _ in range(100)]
    rewards_per_episode = []
    nb_steps = 0
    action_counts = {a: 0 for a in range(action_space.n)}
    done_count = 0
    done_per_episode = []
    for episode in range(episodes):
        if momentum == 0:
            break

        # Initialisation
        observation, info = env.reset(seed=training_seed)
        observation = preprocess_observation(observation)
        state = serialize_state(observation)
        action = epsilon_greedy_policy(state, Q, epsilon, action_space)
        rewards_episode = 0

        if episode % 100 == 1:
            batch_reward = [0 for _ in range(100)]
            t1 = time.time()
            done_count = 0
            epsilon = 0.1

        for step in range(max_steps):
            next_observation, reward, done, truncated, agent_pos = env.step(action)
            nb_steps += 1
            action_counts[action] += 1

            if done:
                done_count += 1
            if done_count > 10:
                epsilon = 0.01

            next_state = serialize_state(next_observation)
            next_action = epsilon_greedy_policy(next_state, Q, epsilon, action_space)

            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )

            state = next_state
            action = next_action

            batch_reward[episode % 100] += reward
            rewards_episode += reward

            if done or truncated:
                break

        rewards_per_episode.append(rewards_episode)

        if episode % 100 == 0 and episode != 0:
            t2 = time.time()
            max_reward = max(batch_reward)
            indice = batch_reward.index(max_reward)
            print(f"Episode {episode}, table size: {len(Q)}, nombre de done: {done_count} temps: {t2 - t1}")
            print(f"Max Reward: {max_reward} sur l'episode {episode - 100 + indice}")
            print("-----------------------------------------------------------")
            done_per_episode.append(done_count)
            if done_count >= 90:
                momentum -= 1

    env.close()

    print(pourcent_action_count(action_counts, nb_steps))
    if(plot):
        plot_courbes(len(rewards_per_episode), rewards_per_episode, done_per_episode)
        
    print(f"======= Execution d'un jeu sur la seed {training_seed} =======")
    execute_game_from_table(max_steps, Q, training_seed)
        
        

def sarsa_1_seed_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, training_seed, amplification, save_data=False, plot=False):
    """Execute une sequence d'entrainement pour une seed complete avec SARSA.

    plot_courbes (bool, optional): Est-ce qu'on veut afficher les courbes ou pas. Defaults to False.
    """
    action_space = env.action_space

    momentum = 10  # Nombre de fois qu'on va faire encore 100 episodes lorsqu'on atteint un moment ou le training est "debloque"
    t1 = time.time()
    batch_reward = [0 for _ in range(100)]
    rewards_per_episode = []
    nb_steps = 0
    action_counts = {a: 0 for a in range(action_space.n)}
    done_count = 0
    done_per_episode = []
    for episode in range(episodes):
        if momentum == 0:
            break

        # Initialisation
        observation, info = env.reset(seed=training_seed)
        state = representation(observation)
        action = epsilon_greedy_policy(state, Q, epsilon, action_space)
        rewards_episode = 0

        if episode % 100 == 1:
            batch_reward = [0 for _ in range(100)]
            t1 = time.time()
            done_count = 0
            epsilon = 0.1

        for step in range(max_steps):
            # Interaction avec l'environnement
            next_observation, reward, done, truncated, agent_pos = env.step(action)
            nb_steps += 1
            action_counts[action] += 1

            if done:
                done_count += 1
            if done_count > 10:
                epsilon = 0.01

            next_state = representation(next_observation)
            next_action = epsilon_greedy_policy(next_state, Q, epsilon, action_space)

            # Mettre a jour la Q-valeur (SARSA update)
            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )

            # Passer a l'etat et action suivants
            state = next_state
            action = next_action

            batch_reward[episode % 100] += reward
            rewards_episode += reward

            if done or truncated:
                break

        rewards_per_episode.append(rewards_episode)

        if episode % 100 == 0 and episode != 0:
            t2 = time.time()
            max_reward = max(batch_reward)
            indice = batch_reward.index(max_reward)
            print(f"Episode {episode}, table size: {len(Q)}, nombre de done: {done_count} temps: {t2 - t1}")
            print(f"Max Reward: {max_reward} sur l'episode {episode - 100 + indice}")
            print("-----------------------------------------------------------")
            done_per_episode.append(done_count)
            if done_count >= 90:
                momentum -= 1

    env.close()

    print(pourcent_action_count(action_counts, nb_steps))
    if(plot):
        plot_courbes(len(rewards_per_episode), rewards_per_episode, done_per_episode)
    
    print(f"======= Execution d'un jeu sur la seed {training_seed} =======")
    execute_game_from_table_nv(max_steps, Q, training_seed)



def execute_game_from_table(max_steps, Q, execute_seed):
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode="human")
    #env = OneHotPartialObsWrapper(env)
    action_space = env.action_space

    observation, info = env.reset(seed=execute_seed)
    observation = preprocess_observation(observation)
    state = serialize_state(observation)
    

    print("Début de l'exécution après l'entraînement:")
    env.render()

    for step in range(max_steps):
        env.render()
        action = max(Q[state], key=Q[state].get)
        print(action)
        
        next_observation, reward, done, truncated, info = env.step(action)
        time.sleep(0.1)
        state = serialize_state(next_observation)
        
        if done or truncated:
            print(f"Fini en {step} steps")
            break

    env.close()
    return
        
        
def execute_game_from_table_nv(max_steps, Q, execute_seed):
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode="human")
    #env = OneHotPartialObsWrapper(env)
    action_space = env.action_space

    observation, info = env.reset(seed=execute_seed)
    state = representation(observation)
    

    print("Début de l'exécution après l'entraînement:")
    env.render()

    for step in range(max_steps):
        env.render()
        action = max(Q[state], key=Q[state].get)
        print(action)
        
        next_observation, reward, done, truncated, info = env.step(action)
        time.sleep(0.1)
        state = representation(next_observation)
        
        if done or truncated:
            print(f"Fini en {step} steps")
            break

    env.close()
    return
        

