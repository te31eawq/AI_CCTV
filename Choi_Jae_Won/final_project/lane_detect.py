import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread(r'C:\Users\iot19\Documents\final_project\testlane2.jpg')  # CCTV 화면 파일 경로

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러링으로 노이즈 제거
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny 엣지 검출
edges = cv2.Canny(blurred, 50, 150)

# 허프 변환으로 직선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 가장 긴 하얀색 선분을 찾기 위한 변수 초기화
max_length_white = 0
best_line_white = None

# 가장 긴 초록색 선분을 찾기 위한 변수 초기화
max_length_green = 0
best_line_green = None

# 색상 범위 정의 (초록색 영역)
lower_green = np.array([35, 40, 40])  # 초록색 하한값 (HSV 색 공간)
upper_green = np.array([85, 255, 255])  # 초록색 상한값 (HSV 색 공간)

# 이미지에서 초록색 영역 추출
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR 이미지를 HSV로 변환
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)  # 초록색 마스크 생성

# Canny 엣지 검출 (초록색 영역에 대해서도)
edges_green = cv2.Canny(mask_green, 50, 150)

# 허프 변환으로 초록색 선분 검출
lines_green = cv2.HoughLinesP(edges_green, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 하얀색 선분과 초록색 선분을 각각 검사
for line in lines:
    for x1, y1, x2, y2 in line:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_length_white:
            max_length_white = length
            best_line_white = (x1, y1, x2, y2)

for line in lines_green:
    for x1, y1, x2, y2 in line:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_length_green:
            max_length_green = length
            best_line_green = (x1, y1, x2, y2)

if best_line_white is not None:
    x1, y1, x2, y2 = best_line_white
    cv2.circle(image, (x1, y1), 2, (0, 0, 255), -1)
    cv2.circle(image, (x2, y2), 2, (0, 0, 255), -1)
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

if best_line_green is not None:
    x1, y1, x2, y2 = best_line_green
    cv2.circle(image, (x1, y1), 2, (255, 0, 0), -1)
    cv2.circle(image, (x2, y2), 2, (255, 0, 0), -1)
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 결과 이미지 출력
cv2.imshow('Detected Lanes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
