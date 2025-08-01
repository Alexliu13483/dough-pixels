import cv2
import numpy as np

# === 方法選擇 ===
METHODS = [
    "Original",
    "Histogram Equalization",
    "CLAHE",
    "Laplacian Sharpen",
    "Unsharp Mask",
    "Gamma Correction"
]

def apply_method(img, method_id):
    if method_id == 0:
        return img
    elif method_id == 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    elif method_id == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
    elif method_id == 3:
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharp = cv2.convertScaleAbs(img + 1.0 * laplacian)
        return sharp
    elif method_id == 4:
        blur = cv2.GaussianBlur(img, (9, 9), 10.0)
        unsharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        return unsharp
    elif method_id == 5:
        gamma = 1.5
        look_up = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype('uint8')
        return cv2.LUT(img, look_up)
    return img

# === 主處理函式 ===
def process_image(val=0):
    method_id = cv2.getTrackbarPos("Method", "Dough Processor")
    blur_val = cv2.getTrackbarPos("Blur", "Dough Processor")
    canny1 = cv2.getTrackbarPos("Canny Th1", "Dough Processor")
    canny2 = cv2.getTrackbarPos("Canny Th2", "Dough Processor")

    proc = apply_method(img.copy(), method_id)
    blur = cv2.GaussianBlur(proc, (2 * blur_val + 1, 2 * blur_val + 1), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny1, canny2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(output, [largest], -1, (0, 255, 0), 2)

    combined = np.hstack((proc, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), output))
    cv2.imshow("Dough Processor", combined)

# === 載入圖像 ===
img = cv2.imread('data/sample_dough_image2.jpg')
if img is None:
    raise ValueError("圖檔無法載入，請檢查路徑")

cv2.namedWindow("Dough Processor", cv2.WINDOW_NORMAL)

cv2.createTrackbar("Method", "Dough Processor", 0, len(METHODS) - 1, process_image)
cv2.createTrackbar("Blur", "Dough Processor", 1, 10, process_image)
cv2.createTrackbar("Canny Th1", "Dough Processor", 50, 255, process_image)
cv2.createTrackbar("Canny Th2", "Dough Processor", 150, 255, process_image)

process_image()
cv2.waitKey(0)
cv2.destroyAllWindows()
