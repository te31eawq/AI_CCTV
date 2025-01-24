import cv2
import numpy as np

# 선분 겹침 여부 확인 함수
def is_not_overlapping(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    # 두 선분의 x, y 범위가 겹치는지 체크
    return not (max(x1, x2) < min(x3, x4) or max(x3, x4) < min(x1, x2))

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

# 가장 긴 하얀색 선분 찾기
for line in lines:
    for x1, y1, x2, y2 in line:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_length_white:
            max_length_white = length
            best_line_white = (x1, y1, x2, y2)

# 초록색 선분 검사 (두 차선으로 나누기 위해 각 선분의 길이를 측정)
green_lines = []
for line in lines_green:
    for x1, y1, x2, y2 in line:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        green_lines.append(((x1, y1, x2, y2), length))

# 길이가 긴 순서대로 초록색 선분 정렬
green_lines.sort(key=lambda x: x[1], reverse=True)

# 여러 선분을 수동으로 선택해서 그리기
# 예시로 첫 번째, 두 번째, 네 번째 선분을 선택했다고 가정
selected_lines = [green_lines[0][0], green_lines[5][0]]  # 인덱스를 바꿔 원하는 선분 선택

# 선택한 선분 그리기
for line in selected_lines:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 초록색으로 선분 그리기

# 가장 긴 하얀색 선분 그리기
if best_line_white is not None:
    x1, y1, x2, y2 = best_line_white
    cv2.circle(image, (x1, y1), 2, (0, 0, 255), -1)
    cv2.circle(image, (x2, y2), 2, (0, 0, 255), -1)
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 초록색 엣지 검출 결과 화면에 표시
cv2.imshow('Green Edges', edges_green)

# 결과 이미지 출력
cv2.imshow('Detected Lanes with Selected Green Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
