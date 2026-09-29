import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113

# Pas de quantification
N = 2
pas = 256//N

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

    # Creating a window to display some images
    cv.namedWindow("Image")
	# Displaying the loaded image in the named window
    cv.imshow("Original image", im)

    # Histrogramme de l'image originale
    histogramme = cv.calcHist([im], [0], None, [256], [0, 256])
    plt.figure("Histogramme original")
    plt.plot(histogramme, color="black")
    plt.xlim([0, 256])
    plt.xlabel("Niveau de gris")
    plt.ylabel("Nombre de pixels")
    plt.title("Histogramme de l'image originale")
    plt.grid(True)

    # Quantification de l'image sur N niveaux de gris
    im_quant = np.floor_divide(im, pas) * pas
    # Affichage de l'image quantifiée
    cv.imshow("Image quantifiée", im_quant)

    # Histrogramme de l'image quantifiée
    histogramme_quant = cv.calcHist([im_quant], [0], None, [256], [0, 256])
    plt.figure("Histogramme quantifié")
    plt.plot(histogramme_quant, color="black")
    plt.xlim([0, 256])
    plt.xlabel("Niveau de gris")
    plt.ylabel("Nombre de pixels")
    plt.title("Histogramme de l'image quantifiée")
    plt.grid(True)
    plt.show()

	# Waiting for the user to press ESCAPE before exiting the application	
    key = 0
   
    while key != ESC_KEY and key!= Q_KEY :
        key = cv.waitKey(1)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()