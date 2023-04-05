from matplotlib import pyplot as plt 
import numpy as np 

output_dir = '/home/jhan/6103/AI6103-Group-Project/output/2023-04-05-05-45/bedroom_1x256x256.npz'
output = np.load(output_dir)

fig = output['arr_0']

plt.figure()
plt.imshow(fig[0])
print('Saving figure..')
plt.savefig('/home/jhan/6103/AI6103-Group-Project/sampled_fig.png')
print('Done')
