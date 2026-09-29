import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./imagesDeTest/peppers-128.png', cv2.IMREAD_GRAYSCALE)

img_nearest = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
img_linear  = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

print("NEAREST :", img_nearest.shape)
print("LINEAR  :", img_linear.shape)

# Zone du zoom (à déplacer sur un bord de poivron si besoin)
y0, y1, x0, x1 = 200, 300, 200, 300

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_nearest, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
plt.title("INTER_NEAREST 512×512")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_linear, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
plt.title("INTER_LINEAR 512×512")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_nearest[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
plt.title("INTER_NEAREST (zoom)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_linear[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
plt.title("INTER_LINEAR (zoom)")
plt.axis('off')

plt.tight_layout()
plt.show()