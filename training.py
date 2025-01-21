import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle
from Minigrid import minigrid


def representation(obs):
    grid = obs['image']
    agent_row = 3
    agent_col = 6
    
    OBJECT_TYPES = {
        6: 'ball',
        5: 'key',
        4: 'door',
        7: 'box',
    }
    
    state_representation = {
        'ball': (float('inf'), 0),
        'key': (float('inf'), 0),
        'door': (float('inf'), 0, 0),
        'box': (float('inf'), 0),
        'wall_in_front': 0
    }
    
    for row in range(7):
        for col in range(7):
            obj_type = grid[row, col, 0]  # Type d'objet (1ère colonne)
            if obj_type in OBJECT_TYPES:
                # Calculer la distance de Manhattan à l'agent
                distance = abs(agent_row - row) + abs(agent_col - col)
                obj_name = OBJECT_TYPES[obj_type]
                if(row < agent_row):
                    dir = 1
                elif(row > agent_row):
                    dir = 2
                else:
                    dir = 0
                
                if(obj_type == 4):
                    obj_state = grid[row, col, 2]
                    state_representation[obj_name] = (distance, obj_state, dir)
                else:
                    state_representation[obj_name] = (distance, dir)
    
    if(grid[agent_row - 1, agent_col, 0] == 2):
        state_representation['wall_in_front'] = 1
    
    direction = obs['direction']
    
    return (direction, tuple([
        state_representation['ball'],
        state_representation['key'],
        state_representation['door'],
        state_representation['box'],
        state_representation['wall_in_front']
    ]))


def preprocess_observation(obs):
    """
    Supprime la dimension de couleur dans l'observation.
    """
    image = obs['image']
    obs_without_color = image[:, :, [0, 2]]
    obs['image'] = obs_without_color
    return obs

def save_Q_and_seed(Q, seed=None, path=None):
    print("sauvegarde des fichiers.........(un peu long)")
    if(path is not None):
        Q_dict = {key: dict(value) for key, value in Q.items()}
        with open(path, "wb") as file:
            pickle.dump(Q_dict, file)
    else:
        path_seed = "./Q-table/seed_nouveau.pkl"
        
        if seed is not None:
            with open(path_seed, "rb") as file:
                tab_seed = pickle.load(file)
            
            tab_seed.append(seed)
            l = len(tab_seed)
            
            with open(path_seed, "wb") as file:
                pickle.dump(tab_seed, file)  

            Q_dict = {key: dict(value) for key, value in Q.items()}
            path_Q = f"./Q-table/q_table_v2.0_{l}.pkl"
            
            with open(path_Q, "wb") as file:
                pickle.dump(Q_dict, file)
        else:
            Q_dict = {key: dict(value) for key, value in Q.items()}
            path_Q = f"./Q-table/q_table_final.pkl"
            
            with open(path_Q, "wb") as file:
                pickle.dump(Q_dict, file)
            
        print("Q-table et seed sauvegardés")
        return tab_seed
    
def plot_courbes(episodes, rewards_per_episode, done_per_episode):
    plt.figure(figsize=(10, 5))
    plt.plot(range(episodes), rewards_per_episode, label="Cumul des récompenses")
    plt.xlabel("Épisode")
    plt.ylabel("Cumul des récompenses")
    plt.title("Cumul des récompenses par épisode")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(done_per_episode)), done_per_episode, label="evolution nombre de done par tranche de 100 episodes")
    plt.xlabel("Tranche")
    plt.ylabel("Nombre de Done")
    plt.title("evolution nombre de done par tranche de 100 episodes")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def serialize_state(observation):
    """Transforme une observation en clé pour la Q-table."""
    direction = observation['direction']
    image = observation['image'].flatten() 
    return (direction, tuple(image))

def pourcent_action_count(action_count, nb_step):
    action_mapping = {
        0: "left",
        1: "right",
        2: "forward",
        3: "pickup",
        4: "drop",
        5: "toggle",
        6: "done"
    }

    for a, c in action_count.items():
        action_count[a] = c * 100 / nb_step

    transformed_action_count = {action_mapping[a]: v for a, v in action_count.items()}
    return transformed_action_count

def epsilon_greedy_policy(state, Q, epsilon, action_space):
    """Politique epsilon-greedy pour choisir une action."""
    if np.random.rand() < epsilon:
        return random.randint(0, action_space.n - 2)  # Action aléatoire
    else:
        return max(Q[state], key=Q[state].get)


def training_1_seed_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, training_seed, amplification, save_data = False, plot = False):
    """execute une sequence d'entrainement pour une seed complète: on s'arrête quand on est sur que 
    le modèle sait aller à la box à tous les coups 

        plot_courbes (bool, optional): est ce qu'on veut afficher les courbes ou pas. Defaults to False.
    """
    action_space = env.action_space
    
    momentum = 13     #Nombre de fois qu'on va faire encore 100 épisodes lorsqu'on atteint un moment ou le training est "débloqué"
    t1 = time.time()
    batch_reward = [0 for _ in range(100)]
    rewards_per_episode = []
    nb_steps = 0
    action_counts = {a: 0 for a in range(action_space.n)}
    done_count = 0
    done_per_episode = []
    epsilon = 0.15
    for episode in range(episodes):
        if(momentum == 0):
            break
        #visited_cells = set()
        observation, info = env.reset(seed=training_seed)
        state = representation(observation)
        rewards_episode = 0
        if episode % 100 == 1:
            batch_reward = [0 for _ in range(100)]
            t1 = time.time()
            done_count = 0
        if momentum == 8:
            epsilon = 0.01

        for step in range(max_steps):
            
            action = epsilon_greedy_policy(state, Q, epsilon, action_space)
            next_observation, reward, done, truncated, agent_pos = env.step(action)
            next_state = representation(next_observation)
            nb_steps += 1
            action_counts[action] += 1
            
            if done:
                done_count += 1

            # Mettre à jour la Q-valeur
            Q[state][action] += alpha * (
                reward + gamma * max(Q[next_state].values()) - Q[state][action]
            )
            
            # Passer à l'état suivant
            state = next_state
            batch_reward[episode % 100] += reward
            rewards_episode += reward
            
            if done or truncated:
                break
        
        rewards_per_episode.append(rewards_episode)
        if episode % 100 == 0 and episode != 0:
            t2 = time.time()
            max_reward = max(batch_reward)
            indice = batch_reward.index(max_reward)
            print(f"Episode {episode}, Q-table size: {len(Q)}, nombre de done: {done_count} temps: {t2-t1}")
            print(f"Max Reward: {max_reward} sur l'épisode {episode - 100 + indice}")
            print("-----------------------------------------------------------")
            done_per_episode.append(done_count)
            if(done_count >= 90):
                momentum -= 1
            
    env.close()

    print(pourcent_action_count(action_counts, nb_steps))
    if(plot):
        plot_courbes(len(rewards_per_episode), rewards_per_episode, done_per_episode)
    
    if save_data:
        tab_seed = save_Q_and_seed(Q, training_seed)
        return Q, tab_seed
    return Q

def training_1_seed(env, Q, episodes, alpha, gamma, epsilon, max_steps, training_seed, amplification, save_data = False, plot = False):
    """execute une sequence d'entrainement pour une seed complète: on s'arrête quand on est sur que 
    le modèle sait aller à la box à tous les coups 

        plot_courbes (bool, optional): est ce qu'on veut afficher les courbes ou pas. Defaults to False.
    """
    action_space = env.action_space
    
    momentum = 10        #Nombre de fois qu'on va faire encore 100 épisodes lorsqu'on atteint un moment ou le training est "débloqué"
    t1 = time.time()
    batch_reward = [0 for _ in range(100)]
    rewards_per_episode = []
    nb_steps = 0
    action_counts = {a: 0 for a in range(action_space.n)}
    done_count = 0
    done_per_episode = []
    for episode in range(episodes):
        if(momentum == 0):
            break
        #visited_cells = set()
        observation, info = env.reset(seed=training_seed)
        observation = preprocess_observation(observation)
        state = serialize_state(observation)
        rewards_episode = 0
        if episode % 100 == 1:
            batch_reward = [0 for _ in range(100)]
            t1 = time.time()
            done_count = 0
            epsilon = 0.1

        for step in range(max_steps):
            
            action = epsilon_greedy_policy(state, Q, epsilon, action_space)
            next_observation, reward, done, truncated, agent_pos = env.step(action)

            nb_steps += 1
            action_counts[action] += 1
            
            if done:
                done_count += 1
            if done_count > 10:
                epsilon = 0.01
                
            next_state = serialize_state(next_observation)

            # Mettre à jour la Q-valeur
            Q[state][action] += alpha * (
                reward + gamma * max(Q[next_state].values()) - Q[state][action]
            )
            
            # Passer à l'état suivant
            state = next_state
            batch_reward[episode % 100] += reward
            rewards_episode += reward
            
            if done or truncated:
                break
        
        rewards_per_episode.append(rewards_episode)
        if episode % 100 == 0 and episode != 0:
            t2 = time.time()
            max_reward = max(batch_reward)
            indice = batch_reward.index(max_reward)
            print(f"Episode {episode}, Q-table size: {len(Q)}, nombre de done: {done_count} temps: {t2-t1}")
            print(f"Max Reward: {max_reward} sur l'épisode {episode - 100 + indice}")
            print("-----------------------------------------------------------")
            done_per_episode.append(done_count)
            if(done_count >= 90):
                momentum -= 1
            
    env.close()

    print(pourcent_action_count(action_counts, nb_steps))
    if(plot):
        plot_courbes(len(rewards_per_episode), rewards_per_episode, done_per_episode)
    
    if save_data:
        tab_seed = save_Q_and_seed(Q, training_seed)
        return Q, tab_seed
    return Q
   

def random_training_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, amplification, save_data = False, plot=False):
    """execute une sequence d'entrainement pour une seed complète: on s'arrête quand on est sur que 
    le modèle sait aller à la box à tous les coups 

        plot_courbes (bool, optional): est ce qu'on veut afficher les courbes ou pas. Defaults to False.
    """
    action_space = env.action_space
    print(f"max steps: {max_steps}")
    
    t1 = time.time()
    batch_reward = [0 for _ in range(100)]
    rewards_per_episode = []
    nb_steps = 0
    action_counts = {a: 0 for a in range(action_space.n)}
    done_count = 0
    done_per_episode = []
    epsilon = 0.05
    for episode in range(episodes):
        #visited_cells = set()
        observation, info = env.reset()
        state = representation(observation)
        rewards_episode = 0
        if episode % 100 == 1:
            """if(done_count > 90):
                seed = random.randint(0,100000)
            else:
                print("continue sur la meme seed....")"""
            batch_reward = [0 for _ in range(100)]
            t1 = time.time()
            done_count = 0
            

        for step in range(max_steps):
            
            action = epsilon_greedy_policy(state, Q, epsilon, action_space)
            next_observation, reward, done, truncated, agent_pos = env.step(action)
            nb_steps += 1
            action_counts[action] += 1
            
            if done:
                done_count += 1
                
            next_state = representation(next_observation)

            # Mettre à jour la Q-valeur
            Q[state][action] += alpha * (
                reward + gamma * max(Q[next_state].values()) - Q[state][action]
            )
            
            # Passer à l'état suivant
            state = next_state
            batch_reward[episode % 100] += reward
            rewards_episode += reward
            
            if done or truncated:
                break
        
        rewards_per_episode.append(rewards_episode)
        if episode % 100 == 0 and episode != 0:
            t2 = time.time()
            max_reward = max(batch_reward)
            indice = batch_reward.index(max_reward)
            print(f"Episode {episode}, Q-table size: {len(Q)}, nombre de done: {done_count} temps: {t2-t1}")
            print(f"Max Reward: {max_reward} sur l'épisode {episode - 100 + indice}")
            print("-----------------------------------------------------------")
            done_per_episode.append(done_count)
            
    env.close()

    print(pourcent_action_count(action_counts, nb_steps))
    print(f"Winrate moyen: {sum(done_per_episode) / len(done_per_episode)}")
    if plot:
        plot_courbes(episodes, rewards_per_episode, done_per_episode)
    
    if save_data:
        save_Q_and_seed(Q, path="./Q-table/q_table_v2.0_final.pkl")
        return Q
    
def random_training(env, Q, episodes, alpha, gamma, epsilon, max_steps, amplification, save_data = False):
    """execute une sequence d'entrainement pour une seed complète: on s'arrête quand on est sur que 
    le modèle sait aller à la box à tous les coups 

        plot_courbes (bool, optional): est ce qu'on veut afficher les courbes ou pas. Defaults to False.
    """
    action_space = env.action_space
    print(f"max steps: {max_steps}")
    
    t1 = time.time()
    batch_reward = [0 for _ in range(100)]
    rewards_per_episode = []
    nb_steps = 0
    action_counts = {a: 0 for a in range(action_space.n)}
    done_count = 0
    done_per_episode = []
    for episode in range(episodes):
        #visited_cells = set()
        observation, info = env.reset()
        observation = preprocess_observation(observation)
        state = serialize_state(observation)
        rewards_episode = 0
        if episode % 100 == 1:
            batch_reward = [0 for _ in range(100)]
            t1 = time.time()
            done_count = 0

        for step in range(max_steps):
            
            action = epsilon_greedy_policy(state, Q, epsilon, action_space)
            next_observation, reward, done, truncated, agent_pos = env.step(action)
            """if agent_pos not in visited_cells:
                visited_cells.add(agent_pos)
                reward += 0.09                      #On veut pas totalement annuler le coût du déplacement parce que....... à méditer"""
            nb_steps += 1
            action_counts[action] += 1
            
            if done:
                done_count += 1
                
            next_state = serialize_state(next_observation)

            # Mettre à jour la Q-valeur
            Q[state][action] += alpha * (
                reward + gamma * max(Q[next_state].values()) - Q[state][action]
            )
            
            # Passer à l'état suivant
            state = next_state
            batch_reward[episode % 100] += reward
            rewards_episode += reward
            
            if done or truncated:
                break
        
        rewards_per_episode.append(rewards_episode)
        if episode % 100 == 0 and episode != 0:
            t2 = time.time()
            max_reward = max(batch_reward)
            indice = batch_reward.index(max_reward)
            print(f"Episode {episode}, Q-table size: {len(Q)}, nombre de done: {done_count} temps: {t2-t1}")
            print(f"Max Reward: {max_reward} sur l'épisode {episode - 100 + indice}")
            print("-----------------------------------------------------------")
            done_per_episode.append(done_count)
            
    env.close()

    print(pourcent_action_count(action_counts, nb_steps))
    
    if save_data:
        tab_seed = save_Q_and_seed(Q)
        return Q, tab_seed