from matplotlib import pyplot as plt 
import numpy as np 

img_dir = '/home/jhan/6103/AI6103-Group-Project/output/2023-04-05-05-45/bedroom_1x256x256.npz'
save_dir = '/home/jhan/6103/AI6103-Group-Project/sampled_fig.png'
img_num = 0


imgs = np.load(img_dir)
fig = imgs['arr_0']
plt.figure()
plt.imshow(fig[img_num])
print('Saving figure..')
plt.savefig(save_dir)
print('Done')
