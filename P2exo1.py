import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./imagesDeTest/peppers-512.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

assert img is not None, "file could not be read, check with os.path.exists()"

k = []
k.append((1/16) * np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]))   # K1
k.append((1/9) * np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]))    # K2
k.append(np.array([[ 1, -3,  1],
                   [-3,  9, -3],
                   [ 1, -3,  1]]))         # K3
k.append(np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]))           # K4
k.append(np.array([[0, -1, -1],
                   [1,  0, -1],
                   [1,  1,  0]]))          # K5
k.append(np.array([[0,  1, 0],
                   [1, -4, 1],
                   [0,  1, 0]]))           # K6

def apply_kernel(img, kernel):
    return cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kernel.astype(np.float32))

res = []
for i in k:
    res.append(apply_kernel(img, i))
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img, cmap="gray")
plt.title("Image originale")
plt.axis("off")
plt.show()

for i in range(len(res)):        
    plt.figure()
    plt.imshow(res[i], cmap="gray")
    plt.title(f"K{i+1}")       
    plt.axis("off")
    plt.show()