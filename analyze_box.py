import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# 1. 이미지 로드 및 그레이스케일 변환
img = cv2.imread('box.jpg') # 사진 파일명에 맞게 수정
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 전처리: 메디안 필터 (바닥의 자갈 노이즈 제거용)
# 수업에서 배운 핵심 내용! 노이즈를 줄여야 가짜 에지가 안 생깁니다.
denoised = cv2.medianBlur(gray, 7) 

# 3. Canny 에지 검출
# 임계값(Threshold)을 조절하며 파손 부위가 잘 나오는지 확인합니다.
edges = cv2.Canny(denoised, 50, 150)

# 4. 결과 시각화
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Pre-processed (Median Filter)")
plt.imshow(denoised, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('result.png') # 결과를 이미지로 저장
plt.show()

print("분석 완료! 'result.png' 파일을 확인하세요.")
