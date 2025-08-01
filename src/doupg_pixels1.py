import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# 載入影像
image = cv2.imread("data/sample_dough_image2.jpg")

# Step 1: 模糊處理
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: 灰階轉換
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# Step 3: 邊緣偵測
edges = cv2.Canny(gray, 50, 210)

# Step 4: 邊緣擴張
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Step 5: 尋找輪廓
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找出最大輪廓（假設瓶子是最大的連通區）
bottle_contour = max(contours, key=cv2.contourArea)

# Step 6: 建立遮罩並填滿輪廓
mask = np.zeros_like(gray)
cv2.drawContours(mask, bottle_contour, -1, color=255, thickness=cv2.FILLED)

# Step 7: 統計白色像素數（即麵團區域）
dough_area_pixels = np.sum(mask == 255)

# Step 8: 在原圖上畫出麵團區域（輪廓）
annotated_img = image.copy()
cv2.drawContours(annotated_img, bottle_contour, -1, (0, 0, 255), thickness=2)  # 紅色輪廓

# Step 9: 使用 PIL + DejaVu 字型疊加文字
image_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)

# 載入 DejaVuSans 字型（請確認路徑存在）
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(font_path, 32)

draw.text((50, 50), f"Dough Area: {dough_area_pixels} pixels", font=font, fill=(255, 0, 0))

# 轉回 OpenCV 格式
final_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# 顯示結果
cv2.imshow("Dough Area Estimation", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
