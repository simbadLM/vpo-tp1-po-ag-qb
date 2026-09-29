import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Chargement image en niveaux de gris
img_gray = cv2.imread("./imagesDeTest/peppers-512.png", cv2.IMREAD_GRAYSCALE)
print(f"Forme (gris) : {img_gray.shape}, Type : {img_gray.dtype}")

#hauteur et largeur 
h = img_gray.shape[0]
w = img_gray.shape[1]

# 2. Définir le centre de rotation et l'angle
centre = (w // 2, h // 2)
angle = 15.0  # Rotation de 15 degrés dans le sens inverse des aiguilles d'une montre
echelle = 1.0 # Conserver la même taille

# 3. Calculer la matrice de rotation
matrice_rotation= cv2.getRotationMatrix2D(centre, angle, echelle)

# 4. Appliquer la rotation avec l'interpolation INTER_LINEAR
image_rotation = cv2.warpAffine(
    img_gray, 
    matrice_rotation, 
    (w, h), 
    flags=cv2.INTER_LINEAR
)

for i in range(5):
    # 4. Appliquer la rotation avec l'interpolation INTER_LINEAR
    image_rotation = cv2.warpAffine(
        image_rotation, 
        matrice_rotation, 
        (w, h), 
        flags=cv2.INTER_LINEAR
    )


#affichage img avant rotation 
cv2.imshow('image avant',img_gray)
cv2.waitKey(0)

#affichage image après rotation
cv2.imshow('image après',image_rotation)
cv2.waitKey(0)

plt.show()






