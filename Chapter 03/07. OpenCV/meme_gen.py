import cv2
import numpy as np
import matplotlib.pyplot as plt

def meme_gen(image, text_up=str, text_down=str):
    img = cv2.imread(image)
    meme = cv2.putText(img, text_up, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 255, 255), 5)
    plt.imshow(meme)

meme_gen('/Users/szokirov/Downloads/168304489956390.png', 'Hello')

