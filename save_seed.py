import pickle
import random

path_seed = "./Q-table/seed_nouveau.pkl"

with open(path_seed, "rb") as file:
        tab_seed = pickle.load(file)


print(tab_seed)
print(len(tab_seed))

"""tab_seed = [42]

with open(path_seed, "wb") as file:
        pickle.dump(tab_seed, file)"""

"""with open(path_seed, "rb") as file:
        tab_seed = pickle.load(file)

print(tab_seed)
tab_seed.append(random.randint(0,100))
print(tab_seed)
        
with open(path_seed, "wb") as file:
        pickle.dump(tab_seed, file)

with open(path_seed, "rb") as file:
        tab_seed = pickle.load(file)
print(tab_seed)"""

