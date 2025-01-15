import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import minigrid
import random
import pickle
import time

from collections import defaultdict
from minigrid.wrappers import OneHotPartialObsWrapper
from training import training_1_seed, training_1_seed_nv, random_training, random_training_nv
from training_sarsa import sarsa_1_seed
from execute_game import execute_game, execute_game_nv

SAVE = False
LOAD = False
INIT_TRAINING = True
SARSA = True

# Paramètres d'apprentissage
alpha = 0.1  # Taux d'apprentissage
gamma = 0.99  # Facteur de discount
epsilon = 0.1  # Exploration epsilon-greedy
episodes = 5000
max_steps = 500  # Étapes par épisode

amplification = 10   #facteur d'amplification pour la formule de reward (plus il est grand plus la diminution du nombre de pas est privilégié)

pretrained_seed_path = "./Q-table/seed_nouveau.pkl"



def preprocess_observation(obs):
    """
    Supprime la dimension de couleur dans l'observation.
    """
    # Extraire la grille de l'observation
    image = obs['image']
    # Supprimer la dimension couleur (index 1) et garder type (0) et état (2)
    obs_without_color = image[:, :, [0, 2]]
    # Mettre à jour l'observation avec la grille pré-traitée
    obs['image'] = obs_without_color
    return obs

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

def pourcent_action_count(action_count, nb_step):
    for a, c in action_count.items():
        action_count[a] = c * 100 / nb_step
    return action_count

def serialize_state(observation):
    """Transforme une observation en clé pour la Q-table."""
    direction = observation['direction']
    image = observation['image'].flatten() 
    return (direction, tuple(image))

def epsilon_greedy_policy(state, Q, epsilon, action_space):
    """Politique epsilon-greedy pour choisir une action."""
    if np.random.rand() < epsilon:
        return random.randint(0, action_space.n - 2)  # Action aléatoire
    else:
        return max(Q[state], key=Q[state].get)
    


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
            Q, tab_seed = training_1_seed_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, seed, amplification, save_data=SAVE, plot_courbes=False)
            
            #print(f"======= Execution d'un jeu sur la seed {seed} =======")
            #l = len(tab_seed)
            #execute_game_nv(max_steps, f"./Q-table/q_table_v2.0_{l}.pkl", seed)
            #print(f"Le modèle a été entrainé sur {l} environnements différents, les voici: {tab_seed}")
        else:
            training_1_seed(env, Q, episodes, alpha, gamma, epsilon, max_steps, seed, amplification, save_data=SAVE, plot_courbes=False)
    else:
        tab_seed, Q = load_data(action_space, final=True)
        print(f"======= Début de l'entrainement aléatoire ========")
        print(f"longueur initial de Q: {len(Q)}")
        Q = random_training_nv(env, Q, episodes, alpha, gamma, epsilon, max_steps, amplification, save_data=SAVE, plot_courbes=False)
else:
    tab_seed, Q = load_data(action_space)
    seed = random.randint(0,10000)
    while seed in tab_seed:
        seed = random.randint(0,10000)
    print(f"longueur initiale de la Q-table: {len(Q)}")
    print(f"======= Début de l'entrainement avec SARSA sur la seed {seed} ========")
    
    sarsa_1_seed(env, Q, episodes, alpha, gamma, epsilon, max_steps, seed, amplification, save_data=SAVE, plot_courbes=False)
