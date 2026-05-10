import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 로드
img = cv2.imread('box.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 전처리 강화: Bilateral Filter 
# (시그마 값이 커질수록 노이즈 제거가 강력해지지만 연산이 조금 더 걸립니다)
# d=9, sigmaColor=75, sigmaSpace=75 가 표준적입니다.
denoised = cv2.bilateralFilter(gray, 9, 100, 100)

# 3. Canny 에지 검출 (임계값을 조금 더 높여서 약한 자갈 에지를 무시합니다)
edges = cv2.Canny(denoised, 80, 160)

# 4. 형태학적 연산 (Morphology): 작은 에지 파편 제거
# 자갈처럼 작은 점들은 사라지게 만듭니다.
kernel = np.ones((3,3), np.uint8)
# Opening은 침식(Erosion) 후 팽창(Dilation)으로 작은 점들을 지웁니다.
cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

# 5. 결과 시각화
plt.figure(figsize=(15, 10))

titles = ["Original", "Bilateral Filtered", "Cleaned Edges"]
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), denoised, cleaned_edges]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray' if i>0 else None)
    plt.axis('off')

plt.tight_layout()
plt.savefig('result_v2.png')
plt.show()

print("업그레이드 분석 완료! 'result_v2.png'를 확인하세요.")
