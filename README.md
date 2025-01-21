Pour les entrainements, le code va automatiquement lancer le jeu sur la seed utilisée pour l'entrainement

Pour executer un entrainement simple(Q-learning avec ancien et nouveau cerveau):

TRAINING = True
SAVE = False
LOAD = False
INIT_TRAINING = True
SARSA = False

EXECUTE_GAME = False

=============================

Pour executer un entrainement simple(Sarsa ET Q-learning):

TRAINING = True
SAVE = False
LOAD = False
INIT_TRAINING = True
SARSA = True

EXECUTE_GAME = False


=============================

Pour lancer le jeu sur une seed aléatoire jamais vu par l'agent:

TRAINING = False
SAVE = False
LOAD = False
INIT_TRAINING = True
SARSA = False

EXECUTE_GAME = True

=============================

Pour executer un entrainement aléatoire:

TRAINING = True
SAVE = False
LOAD = True
INIT_TRAINING = False
SARSA = False

EXECUTE_GAME = False

