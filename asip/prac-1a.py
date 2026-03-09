import cv2
import numpy as np 
import matplotlib.pyplot as plt
image=cv2.imread("C:\\Users\\abhay\\asip\\grayscaleImage.jpg")
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image,cmap="gray")

upsample_image=cv2.resize(image,None,fx=2,fy=2,interpolation =cv2.INTER_LINEAR)
plt.subplot(1,3,2)
plt.title("Upsampled Image")
plt.imshow(upsample_image,cmap="gray")

downsample_image=cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
plt.subplot(1,3,3)
plt.title("Downsampled Image")
plt.imshow(downsample_image, cmap="gray")

plt.tight_layout()
plt.show()