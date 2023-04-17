from matplotlib import pyplot as plt 
import numpy as np 
import os

img_dir = '/home/jhan/6103/AI6103-Group-Project/output/2023-03-29-02-21/samples_10000x64x64x3.npz'
save_dir = '/home/jhan/6103/AI6103-Group-Project/sampled_images/64_by_64_imagenet'

imgs = np.load(img_dir)
fig = imgs['arr_0']
for i in range(100):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, 'clip_fig_{}.png'.format(i))
    plt.figure()
    plt.imshow(fig[i])
    print('Saving figure {}..'.format(i+1))
    plt.savefig(filename)
print('Done')
