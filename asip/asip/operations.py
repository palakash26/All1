import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Step 1: Load grayscale image
# -------------------------
img = cv2.imread("C:\\Users\\abhay\\asip\\grayscaleImage.jpg", cv2.IMREAD_GRAYSCALE)

# -------------------------
# 1. Log Transformation
# -------------------------
c = 255 / np.log(1 + np.max(img))
log_transformed = c * np.log(1 + img.astype(np.float64))
log_transformed = np.array(log_transformed, dtype=np.uint8)

# -------------------------
# 2. Power-Law (Gamma) Transformation
# -------------------------
gamma = 0.5  # gamma < 1 brightens the image
power_law = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)

# -------------------------
# 3. Contrast Adjustment (Linear Stretching)
# -------------------------
a = 1.5  # gain
b = 0    # bias
contrast_img = cv2.convertScaleAbs(img, alpha=a, beta=b)

# -------------------------
# 4. Histogram Equalization
# -------------------------
hist_eq = cv2.equalizeHist(img)

# -------------------------
# 5. Thresholding
# -------------------------
# Simple binary threshold
_, thresh_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# -------------------------
# 6. Halftoning (Simple Dithering)
# -------------------------
# Create a checkerboard halftone by thresholding with a pattern
rows, cols = img.shape
halftone = np.zeros_like(img)
for i in range(rows):
    for j in range(cols):
        if (i+j) % 2 == 0:
            halftone[i,j] = 255 if img[i,j] > 127 else 0
        else:
            halftone[i,j] = 0 if img[i,j] > 127 else 255

# -------------------------
# Step 7: Display Results
# -------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(2,3,2)
plt.title("Log Transformation")
plt.imshow(log_transformed, cmap='gray')

plt.subplot(2,3,3)
plt.title("Power-Law (Gamma)")
plt.imshow(power_law, cmap='gray')

plt.subplot(2,3,4)
plt.title("Contrast Adjusted")
plt.imshow(contrast_img, cmap='gray')

plt.subplot(2,3,5)
plt.title("Histogram Equalization")
plt.imshow(hist_eq, cmap='gray')

plt.subplot(2,3,6)
plt.title("Thresholding & Halftone")
plt.imshow(halftone, cmap='gray')

plt.tight_layout()
plt.show()
