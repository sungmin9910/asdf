import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 로드
img = cv2.imread('box.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 전처리: 강한 블러로 박스 형태만 남기기
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# 3. 이진화(Thresholding): 박스와 배경 분리
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. 윤곽선 찾기 및 마스크 생성 (가장 큰 사각형인 '박스'만 추출)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# 5. 마스크 적용: 박스 영역 외의 자갈 바닥은 모두 검은색으로 처리
masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

# 6. 마스킹된 영역 안에서만 Canny 에지 검출
# 배경이 없으므로 자갈 에지가 생기지 않습니다!
final_edges = cv2.Canny(masked_gray, 50, 150)

# 7. 결과 시각화
plt.figure(figsize=(15, 10))
titles = ["Original", "Box Masking", "Final Edges (No Background)"]
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask, final_edges]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray' if i>0 else None)
    plt.axis('off')

plt.tight_layout()
plt.savefig('result_final.png')
plt.show()

print("완벽 분석 완료! 'result_final.png'를 확인하세요.")
