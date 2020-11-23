from data_init import DPR_dataset

test_dataset = DPR_dataset()
print(len(test_dataset))
image, target = test_dataset[139]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
xy = target["boxes"]
# showROI(rotated_image, rotated_xy)
fig, ax = plt.subplots(1)
ax.imshow(image.transpose(0,2).transpose(0,1), cmap='gray')
for k in range(8):
    w = xy[k][2] - xy[k][0]
    h = xy[k][3] - xy[k][1]
    xy_coord = xy[k][:2]
    rect = patches.Rectangle(xy_coord, 50 , 50, linewidth=1,edgecolor='r',fill=False)
    ax.add_patch(rect)
plt.show()


