from matplotlib import pyplot as plt 
import numpy as np 

output_dir = '/home/jhan/6103/AI6103-Group-Project/output/2023-03-28-07-57/samples_100x64x64x3.npz'
output = np.load(output_dir)

fig = output['arr_0']
class_num = output['arr_1']

plt.figure()
plt.title(class_num[20])
plt.imshow(fig[20])
plt.savefig('output.png')
