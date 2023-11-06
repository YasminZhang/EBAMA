import matplotlib.pyplot as plt

# Data for animals and objects
data_animals = {'lambda': [0, 0.25, 0.5, 0.75, 1.0], 'Full Clip': [0.339, 0.341, 0.341, 0.341, 0.339]}
data_objects = {'lambda': [0, 0.25, 0.5, 0.75, 1.0], 'Full Clip': [0.361, 0.367, 0.368, 0.367, 0.366]}
data_animals_objects = {'lambda': [0, 0.25, 0.5, 0.75, 1.0], 'Full Clip': [0.363,0.363,0.361,0.36,0.359]}

# Plotting
plt.figure(figsize=(10, 5))
plt.plot('lambda', 'Full Clip', data=data_animals, marker='o', color='blue', label='Animal-Animal')
plt.plot('lambda', 'Full Clip', data=data_animals_objects, marker='o', color='red', label='Animal-Object')
plt.plot('lambda', 'Full Clip', data=data_objects, marker='o', color='green', label='Object-Object')

# Labeling
plt.title('Text-Image Full Similarity with different $\lambda$', fontsize=20)
# let xlabel be $\lambda$, larger font size
plt.xlabel('$\lambda$', fontsize=20)
plt.ylabel('Text-Image Full Similarity', fontsize=20)
# let x axis be 0. 0.25, 0.5, 0.75, 1.0
plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.legend()
plt.grid(True)
plt.savefig('./plots/full_clip.png')