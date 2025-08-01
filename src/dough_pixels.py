import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
image = cv2.imread('data/sample_dough_image.jpg')

# 將 BGR 轉為 RGB（方便用 matplotlib 顯示）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 顯示圖片
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# 轉灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 模糊處理以去除雜訊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 邊緣偵測
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# 尋找輪廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找出最大輪廓（假設瓶子是最大的連通區）
bottle_contour = max(contours, key=cv2.contourArea)

# 建立與圖片同尺寸的黑色遮罩
bottle_mask = np.zeros_like(gray)

# 在遮罩上畫出瓶子輪廓並填滿
cv2.drawContours(bottle_mask, [bottle_contour], -1, 255, thickness=cv2.FILLED)

# 顯示瓶子遮罩
plt.imshow(bottle_mask, cmap='gray')
plt.title('Bottle Mask')
plt.axis('off')
plt.show()

# 轉 HSV 色彩空間
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義淺色（白色/奶白）範圍
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 80, 255])

# 取得麵糰遮罩
dough_mask = cv2.inRange(hsv, lower_white, upper_white)

# 顯示麵糰遮罩
plt.imshow(dough_mask, cmap='gray')
plt.title('Dough Mask')
plt.axis('off')
plt.show()


