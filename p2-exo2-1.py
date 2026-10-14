import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113

def erosion(image):
    image_erodee = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            tous_blancs = True
            for yp in range(-1, 2):
                for xp in range(-1, 2):
                    voisin_y = y + yp
                    voisin_x = x + xp
                    if (voisin_y < 0 or voisin_y >= image.shape[0] # Si un pixel est noir ou si on touche un pixel extérieur
                            or voisin_x < 0 or voisin_x >= image.shape[1]
                            or image[voisin_y, voisin_x] == 0):
                        tous_blancs = False

            if tous_blancs:
                image_erodee[y, x] = 255

    return image_erodee

def dilatation(image):
    image_dilatation= np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            tous_blancs = False
            for yp in range(-1, 2):
                for xp in range(-1, 2):
                    voisin_y = y + yp
                    voisin_x = x + xp
                    if (0 <= voisin_y < image.shape[0]
                            and 0 <= voisin_x < image.shape[1]
                            and image[voisin_y, voisin_x] == 255):
                        tous_blancs = True

            if tous_blancs:
                image_dilatation[y, x] = 255

    return image_dilatation

def main():
    # Data structure to store the image
    im = None
    # default name of the image file
    imName= "./imagesDeTest/peppers-512.png"
   
    # If we give an argument then open it instead of the default image
    if len(sys.argv) == 2 :
      imName = sys.argv[1]
   
	# Reading the image (and forcing it to grayscale)
    print("reading image")
    im = cv.imread(imName,cv.IMREAD_GRAYSCALE)

    if im is None or im.size == 0 or (im.shape[0] == 0) or (im.shape[1] == 0):
        print("Could not load image !")
        print("Exiting now...")
        exit(1)

    # Binaire de l'image
    _, binary = cv.threshold(im, 128, 255, cv.THRESH_BINARY)

    # Affichage de l'image binaire
    cv.imshow("Image binaire", binary)

    im_erosion = erosion(binary)
    cv.imshow("Image érodée", im_erosion)

    im_dilatation = dilatation(binary)
    cv.imshow("Image dilatée", im_dilatation)

    # Ouverture : erosion puis dilatation
    im_ouverture = dilatation(erosion(binary))
    # cv.imshow("Ouverture", im_ouverture)

    # Fermeture : dilatation puis erosion
    im_fermeture = erosion(dilatation(binary))
    # cv.imshow("Fermeture", im_fermeture)

    # Gradient morphologique
    im_gradient = im_dilatation - im_erosion
    # cv.imshow("Gradient morphologique", im_gradient)

    # Comparaison avec OpenCV
    elem_struct = np.ones((3, 3), np.uint8)
    open_erosion = cv.erode(binary, elem_struct,
                             borderType=cv.BORDER_CONSTANT,
                             borderValue=0)
    cv.imshow("OpenCV érosion", open_erosion)

    open_dilatation = cv.dilate(binary, elem_struct,
                                 borderType=cv.BORDER_CONSTANT,
                                 borderValue=0)
    cv.imshow("OpenCV dilatation", open_dilatation)

    erosion_difference = np.count_nonzero(im_erosion != open_erosion)
    print("Nombre de pixels différents sur l'érosion :", erosion_difference)

    dilatation_difference = np.count_nonzero(im_dilatation != open_dilatation)
    print("Nombre de pixels différents sur la dilatation :", dilatation_difference)

	# Waiting for the user to press ESCAPE before exiting the application	
    key = 0
   
    while key != ESC_KEY and key!= Q_KEY :
        key = cv.waitKey(1)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()