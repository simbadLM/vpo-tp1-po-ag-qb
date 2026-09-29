import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Chargement en niveaux de gris
img_gray = cv2.imread("./imagesDeTest/peppers-512.png", cv2.IMREAD_GRAYSCALE)
print(f"Forme (gris) : {img_gray.shape}, Type : {img_gray.dtype}")
# Forme typique : (H, W)

hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

#Afficher l'histogramme avec Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(hist, color='black')
plt.title("Histogramme de l'image (Niveaux de gris)")
plt.xlabel("Intensité des pixels (0 - 255)")
plt.ylabel("Nombre de pixels")
plt.xlim([0, 256])
plt.grid(True)
plt.show()
