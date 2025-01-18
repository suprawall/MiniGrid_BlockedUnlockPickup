import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import time

from Minigrid2 import minigrid
from collections import defaultdict
from training import training_1_seed, training_1_seed_nv, random_training, random_training_nv
from training_sarsa import sarsa_1_seed, sarsa_1_seed_nv, execute_game_from_table, execute_game_from_table_nv
from execute_game import execute_game, execute_game_nv

SAVE = False
LOAD = False
INIT_TRAINING = True
SARSA = False

# Paramètres d'apprentissage
alpha = 0.1  
gamma = 0.99  
epsilon = 0.1  
episodes = 5000
max_steps = 500  

amplification = 10   #facteur d'amplification pour la formule de reward (plus il est grand plus la diminution du nombre de pas est privilégié)

pretrained_seed_path = "./Q-table/seed_nouveau.pkl"


def load_data(action_space, final=False):
    with open(pretrained_seed_path, "rb") as file:
            tab_seed = pickle.load(file)
            
    l = len(tab_seed)
    path_qtable = f"./Q-table/q_table_v2.0_{l}.pkl"
    if LOAD:
        if final == False:
            with open(path_qtable, "rb") as file:
                Q_loaded = pickle.load(file)
            Q = defaultdict(lambda: {a: 0 for a in range(action_space.n - 1)}, Q_loaded)
        else:
            with open("./Q-table/q_table_v2.0_final.pkl", "rb") as file:
                Q_loaded = pickle.load(file)
            Q = defaultdict(lambda: {a: 0 for a in range(action_space.n - 1)}, Q_loaded)
    else:
        Q = defaultdict(lambda: {a: 0 for a in range(action_space.n - 1)})
    
    return tab_seed, Q
    


env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
#env = OneHotPartialObsWrapper(env)
action_space = env.action_space


if not SARSA:
    if INIT_TRAINING:
        tab_seed, Q = load_data(action_space)
        seed = random.randint(0,10000)
        while seed in tab_seed:
            seed = random.randint(0,10000)

        print(f"longueur initiale de la Q-table: {len(Q)}")
        print(f"======= Début de l'entrainement avec Q-learning sur la seed {seed} ========")
        if SAVE:
            Q, tab_seed = training_1_seed_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, seed, amplification, save_data=SAVE, plot=False)
            
            #print(f"======= Execution d'un jeu sur la seed {seed} =======")
            #l = len(tab_seed)
            #execute_game_nv(max_steps, f"./Q-table/q_table_v2.0_{l}.pkl", seed)
            #print(f"Le modèle a été entrainé sur {l} environnements différents, les voici: {tab_seed}")
        else:
            Q = training_1_seed(env, Q, episodes, alpha, gamma, epsilon, max_steps, seed, amplification, save_data=SAVE, plot=True)
            execute_game_from_table(max_steps, Q, seed)
            
            tab_seed, Q = load_data(action_space, final=True)
            print(f"longueur initiale de la Q-table: {len(Q)}")
            print(f"======= Début de l'entrainement avec Q-learning sur la seed {seed} ========")
            Q = training_1_seed_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, seed, amplification, save_data=SAVE, plot=True)
            execute_game_from_table_nv(max_steps, Q, seed)        
    else:
        tab_seed, Q = load_data(action_space, final=True)
        print(f"======= Début de l'entrainement aléatoire ========")
        print(f"longueur initial de Q: {len(Q)}")
        Q = random_training_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, amplification, save_data=SAVE)
        
else:
    tab_seed, Q = load_data(action_space)
    seed = random.randint(0,10000)
    while seed in tab_seed:
        seed = random.randint(0,10000)
    print(f"longueur initiale de la Q-table: {len(Q)}")
    print(f"======= Début de l'entrainement avec SARSA sur la seed {seed} ========")
    
    Q = sarsa_1_seed_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, seed, amplification, save_data=SAVE, plot=True)
    
