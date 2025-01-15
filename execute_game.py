from collections import defaultdict
from minigrid.wrappers import OneHotPartialObsWrapper
#from projet import serialize_state
from training import preprocess_observation, representation

import minigrid
import gymnasium as gym
import pickle
import time
import random

def serialize_state(observation):
    """Transforme une observation en clé pour la Q-table."""
    direction = observation['direction']
    image = observation['image'].flatten() 
    return (direction, tuple(image))


def execute_game_nv(max_steps, path_qtable, execute_seed):
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode="human")
    #env = OneHotPartialObsWrapper(env)
    action_space = env.action_space

    observation, info = env.reset(seed=execute_seed)
    state = representation(observation)
    print(state)
    
    with open(path_qtable, "rb") as file:
        Q_loaded = pickle.load(file)
        Q = defaultdict(lambda: {a: 0 for a in range(action_space.n - 1)}, Q_loaded)
        
    print(f"longueur de Q: {len(Q)}")

    print("Début de l'exécution après l'entraînement:")
    env.render()

    for step in range(max_steps):
        env.render()
        action = max(Q[state], key=Q[state].get)
        print(action)
        
        next_observation, reward, done, truncated, info = env.step(action)
        #time.sleep(0.01)
        next_state = representation(next_observation)
        
        Q[state][action] += 0.1 * (
                reward + 0.99 * max(Q[next_state].values()) - Q[state][action]
        )
        
        state = next_state
        
        if done or truncated:
            print(f"Fini en {step} steps")
            break

    env.close()
    return

def execute_game(max_steps, path_qtable, execute_seed):
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode="human")
    #env = OneHotPartialObsWrapper(env)
    action_space = env.action_space

    observation, info = env.reset(seed=execute_seed)
    observation = preprocess_observation(observation)
    state = serialize_state(observation)
    
    with open(path_qtable, "rb") as file:
        Q_loaded = pickle.load(file)
        Q = defaultdict(lambda: {a: 0 for a in range(action_space.n - 1)}, Q_loaded)

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


#execute_game_nv(75, f"./Q-table/q_table_v2.0_final.pkl", random.randint(1,10000))

"""path_seed = "./Q-table/seed.pkl"

with open(path_seed, "rb") as file:
        tab_seed = pickle.load(file)
    
for seed in tab_seed:
    execute_game(150, f"./Q-table/q_table_{8}.pkl", seed)"""