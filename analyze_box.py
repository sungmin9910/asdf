import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_box(image_path):
    # 이미지 로드 및 그레이스케일
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 전처리: 가우시안 블러 (부드럽게)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 에지 검출: Canny
    # 흰 배경이라 노이즈가 적으므로 임계값을 표준적으로 설정
    edges = cv2.Canny(blurred, 50, 150)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), edges

# 두 이미지 처리
normal_img, normal_edges = process_box('box_or.jpg')
damaged_img, damaged_edges = process_box('box.jpg')

# 결과 시각화
plt.figure(figsize=(12, 10))

# 정상 박스
plt.subplot(2, 2, 1)
plt.title("Normal Box (Original)")
plt.imshow(normal_img)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Normal Box (Edges)")
plt.imshow(normal_edges, cmap='gray')
plt.axis('off')

# 파손 박스
plt.subplot(2, 2, 3)
plt.title("Damaged Box (Original)")
plt.imshow(damaged_img)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Damaged Box (Edges)")
plt.imshow(damaged_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('comparison_result.png')
plt.show()

print("비교 분석 완료! 'comparison_result.png'를 확인하세요.")
