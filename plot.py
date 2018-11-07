"""
Plot focal contrastive loss.
"""
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(121)
# marginal distance for negative pairs.
margin =  1.
# step for ploting curves.
step = 0.01
# sigma control the weight decay rate.
sigma_1 = 0.1
sigma_2 = 0.2
end_point = 1.1
dist = np.arange(0., end_point, step)
offset = margin/2.
def weight_func(input_x, offset, sigma):
    """
    Args:
        input_x: x axis data
        margin: marginal distance for negative pairs.
        sigma:  decay rate for weight func

    Returns:
        the weighted distace
    """

    return  2. / (1. + np.exp(-1. * (input_x - offset) / sigma))

weight = weight_func(dist, offset, sigma_1)
weight_2 = weight_func(dist, offset, sigma_2)

origin_dist = np.square(dist)
weighted_dist =  weight * np.square(dist)
weighted_dist_2 =  weight_2 * np.square(dist)


# plot results
ax.plot(dist, origin_dist, linewidth=2.0, label="$EL$")
ax.plot(dist, weighted_dist, linewidth=2.0, label='$FEL$ $(\sigma=0.1)$')
ax.plot(dist, weighted_dist_2, linewidth=2.0, label='$FEL$ $(\sigma=0.2)$')
# plot split point
ax.plot([0.5]*9, np.arange(0, 0.25, 0.03), 'k--', linewidth=1.5)
ax.text(0.1, 1.5, "$EL=d^2$", fontsize=16)
ax.text(0.1, 1.3, r'$FEL=2*f(\frac{d-h/2}{\sigma})d^2$', fontsize=16)
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
ax.plot(dist, origin_dist_negative, linewidth=2.0, label='$HL$')
ax.plot(dist, weighted_dist_negative, linewidth=2.0, label='$FHL$ $(\sigma=0.1)$')
ax.plot(dist, weighted_dist_negative_2, linewidth=2.0, label='$FHL$ $(\sigma=0.2)$')


ax.plot([0.5]*9, np.arange(0, 0.25, 0.03), 'k--', linewidth=1.5)
ax.text(0.5, 0.02, r'$h/2$', fontsize=12)
ax.text(1., 0.02, r'$h$', fontsize=12)


ax.text(0.7, 1.3, r'$HL=\max\{0, h-d \}^2$', fontsize=16)
ax.text(0.7, 1.1, r'$FHL=2*f(\frac{HL^{0.5}-h/2}{\sigma})*HL$', fontsize=16)
ax.legend(loc='upper right')

ax.set_xlabel("Pairwise distance $d$")
ax.set_ylabel("Loss for negative pair")

plt.show()
