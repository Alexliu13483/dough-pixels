import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_dough_pixels(image_path):
    """
    統計發酵麵糰占據的像素數量
    """
    
    # 步驟1: 載入圖像
    print("步驟1: 載入圖像...")
    img = cv2.imread(image_path)
    if img is None:
        print("錯誤: 無法載入圖像，請檢查路徑是否正確")
        return None
    
    # 轉換為RGB格式（OpenCV預設是BGR）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"圖像尺寸: {img_rgb.shape}")
    
    # 步驟2: 轉換為HSV色彩空間，更容易進行顏色分割
    print("步驟2: 轉換為HSV色彩空間...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 步驟3: 定義麵糰的顏色範圍（白色到淺黃色）
    print("步驟3: 定義麵糰顏色範圍...")
    # HSV範圍：H(色相), S(飽和度), V(明度)
    # 白色麵糰的HSV範圍
    lower_dough = np.array([0, 0, 180])      # 下限
    upper_dough = np.array([30, 50, 255])   # 上限
    
    # 步驟4: 創建遮罩
    print("步驟4: 創建顏色遮罩...")
    mask = cv2.inRange(hsv, lower_dough, upper_dough)
    
    # 步驟5: 形態學操作，清理雜訊
    print("步驟5: 形態學操作清理雜訊...")
    kernel = np.ones((3,3), np.uint8)
    # 開運算：先侵蝕再膨脹，去除小雜訊
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # 閉運算：先膨脹再侵蝕，填補小洞
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 步驟6: 統計像素數量
    print("步驟6: 統計像素數量...")
    dough_pixels = cv2.countNonZero(mask_cleaned)
    total_pixels = img.shape[0] * img.shape[1]
    dough_percentage = (dough_pixels / total_pixels) * 100
    
    # 步驟7: 顯示結果
    print("步驟7: 顯示結果...")
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # HSV image
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    plt.title('HSV Image')
    plt.axis('off')
    
    # Initial mask
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Initial Mask')
    plt.axis('off')
    
    # Cleaned mask
    plt.subplot(1, 4, 4)
    plt.imshow(mask_cleaned, cmap='gray')
    plt.title('Cleaned Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Dough pixels: {dough_pixels:,}")
    print(f"Dough percentage: {dough_percentage:.2f}%")
    
    return {
        'total_pixels': total_pixels,
        'dough_pixels': dough_pixels,
        'dough_percentage': dough_percentage,
        'mask': mask_cleaned
    }

def adjust_color_range(image_path):
    """
    互動式調整顏色範圍的輔助函數
    """
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    def nothing(val):
        pass
    
    # Create window and trackbars
    cv2.namedWindow('HSV Adjustment')
    cv2.createTrackbar('H_min', 'HSV Adjustment', 0, 179, nothing)
    cv2.createTrackbar('S_min', 'HSV Adjustment', 0, 255, nothing)
    cv2.createTrackbar('V_min', 'HSV Adjustment', 180, 255, nothing)
    cv2.createTrackbar('H_max', 'HSV Adjustment', 30, 179, nothing)
    cv2.createTrackbar('S_max', 'HSV Adjustment', 50, 255, nothing)
    cv2.createTrackbar('V_max', 'HSV Adjustment', 255, 255, nothing)
    
    while True:
        # Get trackbar values
        h_min = cv2.getTrackbarPos('H_min', 'HSV Adjustment')
        s_min = cv2.getTrackbarPos('S_min', 'HSV Adjustment')
        v_min = cv2.getTrackbarPos('V_min', 'HSV Adjustment')
        h_max = cv2.getTrackbarPos('H_max', 'HSV Adjustment')
        s_max = cv2.getTrackbarPos('S_max', 'HSV Adjustment')
        v_max = cv2.getTrackbarPos('V_max', 'HSV Adjustment')
        
        # Create mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Show result
        result = cv2.bitwise_and(img, img, mask=mask)
        
        # Display side by side
        display = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
        display = cv2.resize(display, (1200, 400))
        
        cv2.imshow('HSV Adjustment', display)
        
        # Count pixels
        pixels = cv2.countNonZero(mask)
        print(f'\rPixels: {pixels}', end='', flush=True)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"\nOptimal parameters: lower=({h_min}, {s_min}, {v_min}), upper=({h_max}, {s_max}, {v_max})")

# 使用範例
if __name__ == "__main__":
    # 替換為你的圖像路徑
    image_path = "data/sample_dough_image.jpg"
    
    print("Starting dough analysis...")
    result = count_dough_pixels(image_path)
    
    if result:
        print("\nAnalysis completed!")
        print("If results are not accurate, use adjust_color_range() function to fine-tune color range")
        
        # Uncomment the line below for interactive adjustment
        # adjust_color_range(image_path)