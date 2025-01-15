import numpy as np
import random
import time
import pickle

from training import preprocess_observation, serialize_state, pourcent_action_count, epsilon_greedy_policy



def sarsa_1_seed(env, Q, episodes, alpha, gamma, epsilon, max_steps, training_seed, amplification, save_data=False, plot_courbes=False):
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
            # Interaction avec l'environnement
            next_observation, reward, done, truncated, agent_pos = env.step(action)
            nb_steps += 1
            action_counts[action] += 1

            if done:
                done_count += 1
            if done_count > 10:
                epsilon = 0.01

            next_state = serialize_state(next_observation)
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
            print(f"Episode {episode}, Q-table size: {len(Q)}, nombre de done: {done_count} temps: {t2 - t1}")
            print(f"Max Reward: {max_reward} sur l'episode {episode - 100 + indice}")
            print("-----------------------------------------------------------")
            if done_count >= 90:
                momentum -= 1

    env.close()

    print(pourcent_action_count(action_counts, nb_steps))
