import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
image = cv2.imread('sample_dough_image.jpg')

# 將 BGR 轉為 RGB（方便用 matplotlib 顯示）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 顯示圖片
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()
