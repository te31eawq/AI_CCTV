import cv2
import numpy as np

# 선분의 기울기 계산 함수
def get_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

# 선분의 길이 계산 함수
def get_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 선분 겹침 여부 확인 함수
def is_not_overlapping(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    # 두 선분의 x, y 범위가 겹치는지 체크
    return not (max(x1, x2) < min(x3, x4) or max(x3, x4) < min(x1, x2))

# 이미지 불러오기
image = cv2.imread(r'C:\Users\iot19\Documents\final_project\testlane3.jpg')  # CCTV 화면 파일 경로

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러링으로 노이즈 제거
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny 엣지 검출
edges = cv2.Canny(blurred, 50, 150)

# 허프 변환으로 직선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# HSV 색 공간에서 초록색 영역 마스크를 만들어 초록색 차선 찾기
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])  # 초록색의 하한값
upper_green = np.array([85, 255, 255])  # 초록색의 상한값
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)  # 초록색 마스크

# Canny 엣지 검출 (초록색 마스크 영역에 대해서)
edges_green = cv2.Canny(mask_green, 50, 150)

# 허프 변환으로 초록색 선분 검출
lines_green = cv2.HoughLinesP(edges_green, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 초록색 선분을 기울기 기준으로 그룹화
tolerance = 0.1  # 기울기 차이 허용 범위
groups = []

for line in lines_green:
    for x1, y1, x2, y2 in line:
        slope = get_slope(x1, y1, x2, y2)
        added_to_group = False
        for group in groups:
            # 기울기가 비슷한 선분이 있으면 같은 그룹에 추가
            if abs(group[0][4] - slope) < tolerance:
                group.append((x1, y1, x2, y2, slope))
                added_to_group = True
                break
        if not added_to_group:
            groups.append([(x1, y1, x2, y2, slope)])

# 각 그룹에서 가장 긴 선분 하나만 선택
selected_green_lines = []
for group in groups:
    # 길이가 가장 긴 선분 찾기
    longest_line = max(group, key=lambda x: get_length(x[0], x[1], x[2], x[3]))
    selected_green_lines.append(longest_line[:4])  # 선분 좌표만 저장

# 가장 긴 하얀색 선분 찾기
max_length_white = 0
best_line_white = None

for line in lines:
    for x1, y1, x2, y2 in line:
        length = get_length(x1, y1, x2, y2)
        if length > max_length_white:
            max_length_white = length
            best_line_white = (x1, y1, x2, y2)

# 선택된 초록색 선분 그리기
for line in selected_green_lines:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 3)  # 초록색으로 선분 그리기

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
