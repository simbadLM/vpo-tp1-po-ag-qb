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
    imName= "./imagesDeTest/peppers-512-RGB.png"
   
    # If we give an argument then open it instead of the default image
    if len(sys.argv) == 2 :
      imName = sys.argv[1]
   
    print("reading image")
    im = cv.imread(imName, cv.IMREAD_COLOR)

    if im is None or im.size == 0 or (im.shape[0] == 0) or (im.shape[1] == 0):
        print("Could not load image !")
        print("Exiting now...")
        exit(1)

    b, g, r = cv.split(im)
    im_rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Affichage de l'image et des canaux sur une figure
    figure, axes = plt.subplots(2, 3)
    axes[0, 0].imshow(im_rgb)
    axes[0, 0].set_title("Image originale (RGB)")
    axes[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Niveaux de gris")
    axes[0, 2].imshow(r, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Canal R")
    axes[1, 0].imshow(g, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Canal G")
    axes[1, 1].imshow(b, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Canal B")
    axes[1, 2].axis("off")

    for axis in axes.flat:
        axis.axis("off")

    figure.tight_layout()
    plt.show()

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()