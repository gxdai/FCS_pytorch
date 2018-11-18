"""
Plot focal contrastive loss.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(1,1,1)
rec_pre_cl = np.load("cub_results/rec_pre_cl.npy")
rec_pre_fcl = np.load("cub_results/rec_pre_fcl.npy")
rec_pre_tl = np.load("cub_results/rec_pre_tl.npy")
rec_pre_ftl = np.load("cub_results/rec_pre_ftl.npy")
print(rec_pre_cl.shape)
print(rec_pre_fcl.shape)
# CUB 200

"""
NMI_TL = [0.45, 0.45, 0.45, 0.45]
NMI_FTL = [0.50, 0.50, 0.50, 0.50]
"""
# step for ploting curves.
# plot results
markersize=5.0
linewidth=1.5
fillstyle='none'
ax.plot(rec_pre_fcl[:,0], rec_pre_fcl[:,1], '-r', linewidth=linewidth, label="FCL" )
ax.plot(rec_pre_cl[:,0], rec_pre_cl[:,1], '-b', linewidth=linewidth, label="CL" )
ax.plot(rec_pre_ftl[:,0], rec_pre_ftl[:,1], '-m', linewidth=linewidth, label="FTL" )
ax.plot(rec_pre_tl[:,0], rec_pre_tl[:,1], '-k', linewidth=linewidth, label="TL" )
"""
ax.plot(size_set, NMI_FTL, '--bo', markersize=markersize, linewidth=linewidth, label="FTL" )
ax.plot(size_set, NMI_TL, '--gs', markersize=markersize, linewidth=linewidth, label="TL")
"""
# ax.set_ylim(0.4, 0.6)
ax.set_xlim(0., 1.)
ax.legend(loc='upper right')

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

plt.show()
