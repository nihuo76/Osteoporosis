from PIL import Image
from getimagexywh import getimagexywh

def flipping(personname):
    DPR_image, coordinate_8ROI = getimagexywh(personname)
    centered_xy = coordinate_8ROI + 25
    width, _ = DPR_image.size
    centered_xy[:, 0] = width - centered_xy[:, 0]
    flipped_DPR = DPR_image.transpose(Image.FLIP_LEFT_RIGHT)
    centered_xy -= 25
    return flipped_DPR, centered_xy

#
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# flipped_image, flipped_xy = getimagexywh("wangmin")
# # showROI(rotated_image, rotated_xy)
# fig, ax = plt.subplots(1)
# ax.imshow(flipped_image, cmap='gray')
# for k in range(8):
#     rect = patches.Rectangle(flipped_xy[k], 50, 50, linewidth=1,edgecolor='r',fill=False)
#     ax.add_patch(rect)
# plt.show()
# # flipped_xy.show()
