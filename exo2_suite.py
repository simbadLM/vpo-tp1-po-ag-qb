import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Chargement image en niveaux de gris
img_gray = cv2.imread("./imagesDeTest/peppers-512.png", cv2.IMREAD_GRAYSCALE)
print(f"Forme (gris) : {img_gray.shape}, Type : {img_gray.dtype}")

#augmentation du constraste  
alpha = 1.5
beta = 0
resultat = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)

#hist avec image d'origine 
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

#hist avec constraste 
hist_modif = cv2.calcHist([resultat], [0], None, [256], [0, 256])

#Afficher l'histogramme avec Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(hist, color='black')
plt.title("Histogramme de l'image (Niveaux de gris)")
plt.xlabel("Intensité des pixels (0 - 255)")
plt.ylabel("Nombre de pixels")
plt.xlim([0, 256])
plt.grid(True)

#Afficher l'histogramme avec Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(hist_modif, color='black')
plt.title("Histogramme de l'image avec constraste augmentée (Niveaux de gris)")
plt.xlabel("Intensité des pixels (0 - 255)")
plt.ylabel("Nombre de pixels")
plt.xlim([0, 256])
plt.grid(True)

#affichage img avant constraste 
cv2.imshow('image avant',img_gray)
cv2.waitKey(0)

#affichage image après constraste
cv2.imshow('image après',resultat)
cv2.waitKey(0)

plt.show()
