"""
Plot focal contrastive loss.
"""
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(131)
size_set = [64, 128, 256, 512]

# CUB 200
NN_CL = [0.30, 0.34, 0.35, 0.32]
NN_FCL = [0.35, 0.37, 0.36, 0.37]
NMI_CL = [0.51, 0.52, 0.51, 0.50]
NMI_FCL = [0.52, 0.53, 0.53, 0.53]
F1_CL = [0.15, 0.17, 0.16, 0.15]
F1_FCL = [0.16, 0.17, 0.19, 0.19]
mAP_CL = [0.16, 0.18, 0.17, 0.16]
mAP_FCL = [0.18, 0.20, 0.19, 0.19]
"""
NMI_TL = [0.45, 0.45, 0.45, 0.45]
NMI_FTL = [0.50, 0.50, 0.50, 0.50]
"""
# step for ploting curves.
# plot results
markersize=5.0
linewidth=1.5
fillstyle='none'
ax.plot(size_set, NMI_FCL, '--ro', fillstyle=fillstyle, markersize=markersize, linewidth=linewidth, label="FCL" )
ax.plot(size_set, NMI_CL, '--ks', fillstyle=fillstyle, markersize=markersize, linewidth=linewidth, label="CL")
"""
ax.plot(size_set, NMI_FTL, '--bo', markersize=markersize, linewidth=linewidth, label="FTL" )
ax.plot(size_set, NMI_TL, '--gs', markersize=markersize, linewidth=linewidth, label="TL")
"""
ax.set_ylim(0.4, 0.6)
ax.legend(loc='upper left')

ax.set_xlabel("Embedding size")
ax.set_ylabel("NMI score")

ax = plt.subplot(132)

ax.plot(size_set, F1_FCL, '--ro', fillstyle=fillstyle, markersize=markersize, linewidth=linewidth, label="FCL" )
ax.plot(size_set, F1_CL, '--ks', fillstyle=fillstyle, markersize=markersize, linewidth=linewidth, label="CL")
ax.set_ylim(0., 0.25)
ax.legend(loc='upper left')

ax.set_xlabel("Embedding size")
ax.set_ylabel("F1 score")



ax = plt.subplot(133)

ax.plot(size_set, mAP_FCL, '--ro', fillstyle=fillstyle, markersize=markersize, linewidth=linewidth, label="FCL" )
ax.plot(size_set, mAP_CL, '--ks', fillstyle=fillstyle, markersize=markersize, linewidth=linewidth, label="CL")
ax.set_ylim(0., 0.3)
ax.legend(loc='upper left')

ax.set_xlabel("Embedding size")
ax.set_ylabel("mAP")


"""
ax.plot(dist, weighted_dist, linewidth=2.0, label='$J_{fel}$ $(\sigma=0.1)$')
ax.plot(dist, weighted_dist_2, linewidth=2.0, label='$J_{fel}$ $(\sigma=0.2)$')
# plot split point
ax.plot([0.5]*9, np.arange(0, 0.25, 0.03), 'k--', linewidth=1.5)
ax.text(0.1, 1.5, "$J_{el}=d^2$", fontsize=16)
ax.text(0.1, 1.3, r'$J_{fel}=2*f(\frac{d-h/2}{\sigma})d^2$', fontsize=16)
ax.text(0.5, 0.02, r'$h/2$', fontsize=12)
ax.text(1., 0.02, r'$h$', fontsize=12)
ax.legend(loc='upper left')

ax.set_xlabel("Pairwise distance $d$")
ax.set_ylabel("Loss for positive pair")

# for negative loss
ax = plt.subplot(122)
dist_negative = np.maximum(0, margin - dist)

origin_dist_negative = np.square(dist_negative)
weighted_dist_negative = weight_func(dist_negative, offset, sigma_1) * origin_dist_negative
weighted_dist_negative_2 = weight_func(dist_negative, offset, sigma_2) * origin_dist_negative


# plot results
ax.plot(dist, origin_dist_negative, linewidth=2.0, label='$J_{hl}$')
ax.plot(dist, weighted_dist_negative, linewidth=2.0, label='$J_{fhl}$ $(\sigma=0.1)$')
ax.plot(dist, weighted_dist_negative_2, linewidth=2.0, label='$J_{fhl}$ $(\sigma=0.2)$')


ax.plot([0.5]*9, np.arange(0, 0.25, 0.03), 'k--', linewidth=1.5)
ax.text(0.5, 0.02, r'$h/2$', fontsize=12)
ax.text(1., 0.02, r'$h$', fontsize=12)


ax.text(0.7, 1.3, r'$J_{hl}=\max\{0, h-d \}^2$', fontsize=16)
ax.text(0.7, 1.1, r'$J_{fhl}=2*f(\frac{J_{hl}^{0.5}-h/2}{\sigma})*HL$', fontsize=16)
ax.legend(loc='upper right')

ax.set_xlabel("Pairwise distance $d$")
ax.set_ylabel("Loss for negative pair")
"""
plt.show()
