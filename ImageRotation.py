import numpy as np
from getimagexywh import getimagexywh

# Now let's focus on how to rotate the image
# test for github commit
# test of github push

def rotateimage(personname, degree):
    DPR_image, coordinate_8ROI = getimagexywh(personname)
    centered_xy = coordinate_8ROI+25
    rotated_image = DPR_image.rotate(degree, expand=True)
    width, height = DPR_image.size
    centered_xy = np.transpose(centered_xy)
    degree_pi = (degree/180)*np.pi
    rotation_matrix = np.array([[np.cos(degree_pi), -np.sin(degree_pi)], [np.sin(degree_pi), np.cos(degree_pi)]])
    # note that actually the x,y are swapped in the rotation coordinate
    centered_xy = centered_xy[[1, 0], :]
    rotated_xy = np.matmul(rotation_matrix, centered_xy)
    rotated_xy = np.transpose(rotated_xy)
    # recall that we swapped x,y; we need swap them back
    if degree == 90:
        rotated_xy[:, 0] = width + rotated_xy[:, 0]
    elif degree == 180:
        rotated_xy[:, 0] = height + rotated_xy[:, 0]
        rotated_xy[:, 1] = width + rotated_xy[:, 1]
    elif degree == 270:
        rotated_xy[:, 1] = height + rotated_xy[:, 1]
    else:
        pass
    rotated_xy = rotated_xy[:, [1, 0]]
    rotated_xy -= 25
    # finally you need to deduct 25 from both x and y
    return rotated_image, rotated_xy

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# rotated_image, rotated_xy = rotateimage("wangmin",270)
# # showROI(rotated_image, rotated_xy)
# fig, ax = plt.subplots(1)
# ax.imshow(rotated_image, cmap='gray')
# for k in range(8):
#     rect = patches.Rectangle(rotated_xy[k], 50, 50, linewidth=1,edgecolor='r',fill=False)
#     ax.add_patch(rect)
# plt.show()
# rotated_image.show()

