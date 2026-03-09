import cv2
import numpy as np
import matplotlib.pyplot as plt
image=cv2.imread(r"C:\Users\pala6\Downloads\UI-UX\asip\grayscaleImage.jpg")
kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
convolved_Image=cv2.filter2D(image,-1,kernel)
template = cv2.imread(r"C:\Users\pala6\Downloads\UI-UX\asip\image.png", cv2.IMREAD_GRAYSCALE)

# Perform template matching
convolved_Image = cv2.cvtColor(convolved_Image, cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(convolved_Image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Draw rectangle around matched region
top_left = max_loc
h, w = template.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
matched_img = convolved_Image.copy()
cv2.rectangle(matched_img, top_left, bottom_right, 255, 2)

# -------------------------
# Step 4: Display Results
# -------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1,3,2)
plt.title("After Convolution (Edges)")
plt.imshow(convolved_Image, cmap='gray')

plt.subplot(1,3,3)
plt.title("Template Matched")
plt.imshow(matched_img, cmap='gray')

plt.tight_layout()
plt.show()