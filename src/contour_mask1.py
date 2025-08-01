import cv2
import numpy as np

# 圖片路徑
image_path = 'data/sample_dough_image.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"圖片無法讀取：{image_path}")

resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# ----------------------
# 處理方法定義區
# ----------------------

def method1(img):  # 原始灰階 + Canny
    return cv2.Canny(img, th1, th2)

def method2(img):  # 直方圖等化 + Canny
    eq = cv2.equalizeHist(img)
    return cv2.Canny(eq, th1, th2)

def method3(img):  # 對比度增強 + 模糊 + Canny
    enhanced = cv2.convertScaleAbs(img, alpha=contrast/50.0, beta=0)
    blurred = cv2.GaussianBlur(enhanced, (blur*2+1, blur*2+1), 0)
    return cv2.Canny(blurred, th1, th2)

def method4(img):  # 對比度+銳化+模糊+Canny
    enhanced = cv2.convertScaleAbs(img, alpha=contrast/50.0, beta=0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, sharpness, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    blurred = cv2.GaussianBlur(sharpened, (blur*2+1, blur*2+1), 0)
    return cv2.Canny(blurred, th1, th2)

methods = {
    1: method1,
    2: method2,
    3: method3,
    4: method4
}

current_method = 4  # 預設用 Method 4（效果最佳）

# ----------------------
# 初始參數
# ----------------------
blur = 5
contrast = 50
sharpness = 5
th1 = 50
th2 = 210

# ----------------------
# UI Trackbars
# ----------------------
def nothing(x): pass

cv2.namedWindow('Edge Detection UI')
cv2.createTrackbar('Blur', 'Edge Detection UI', blur, 10, nothing)
cv2.createTrackbar('Contrast', 'Edge Detection UI', contrast, 100, nothing)
cv2.createTrackbar('Sharpness', 'Edge Detection UI', sharpness, 10, nothing)
cv2.createTrackbar('Canny Th1', 'Edge Detection UI', th1, 255, nothing)
cv2.createTrackbar('Canny Th2', 'Edge Detection UI', th2, 255, nothing)

print("👉 按 1~4 切換方法 (目前預設 Method 4)")
print("👉 按 q 鍵離開")

while True:
    # 更新參數
    blur = cv2.getTrackbarPos('Blur', 'Edge Detection UI')
    contrast = cv2.getTrackbarPos('Contrast', 'Edge Detection UI')
    sharpness = cv2.getTrackbarPos('Sharpness', 'Edge Detection UI')
    th1 = cv2.getTrackbarPos('Canny Th1', 'Edge Detection UI')
    th2 = cv2.getTrackbarPos('Canny Th2', 'Edge Detection UI')

    # 執行目前方法
    edge = methods[current_method](gray)

    # 偵測輪廓並標示最大輪廓
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = resized.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

    combined = np.hstack((cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR), contour_image))
    cv2.imshow('Edge Detection UI', combined)

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        current_method = int(chr(key))
        print(f"🔄 切換到 Method {current_method}")

cv2.destroyAllWindows()
