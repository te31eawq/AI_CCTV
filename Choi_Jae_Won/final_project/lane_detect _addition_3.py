import cv2
import numpy as np

# 선분의 기울기 계산 함수
def get_slope(x1, y1, x2, y2):
    if (x2 - x1) != 0:
        return (y2 - y1) / (x2 - x1)
    else:
        return float('inf')  # 수직선의 기울기는 'inf'

# 선분의 길이 계산 함수
def get_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 이미지 불러오기
image = cv2.imread(r'C:\Users\iot19\Documents\final_project\testlane4.jpg')  # CCTV 화면 파일 경로

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
tolerance = 0.01  # 기울기 차이 허용 범위를 늘려서 좀 더 유연한 그룹화를 유도
groups = []

for line in lines_green:
    for x1, y1, x2, y2 in line:
        slope = get_slope(x1, y1, x2, y2)
        added_to_group = False
        for group in groups:
            # 기울기 차이가 tolerance 이내이면 같은 그룹에 추가
            if abs(group[0][4] - slope) < tolerance:
                group.append((x1, y1, x2, y2, slope))
                added_to_group = True
                break
        if not added_to_group:
            groups.append([(x1, y1, x2, y2, slope)])

# 그룹 수 및 색상 확인
print(f"Total number of groups: {len(groups)}")
for idx, group in enumerate(groups):
    print(f"Group {idx + 1}: Number of lines = {len(group)}")

# 각 그룹에 대해 다른 색으로 그리기
colors = [
    (0, 0, 255),    # 빨강
    (0, 165, 255),  # 주황
    (0, 255, 255),  # 노랑
    (0, 255, 0),    # 초록
    (255, 0, 0),    # 파랑
    (75, 0, 130),   # 남색
    (238, 130, 238) # 보라
]

for idx, group in enumerate(groups):
    # 각 그룹에서 가장 긴 선분 찾기
    max_length = 0
    longest_line = None
    for line in group:
        x1, y1, x2, y2, slope = line
        length = get_length(x1, y1, x2, y2)
        if length > max_length:
            max_length = length
            longest_line = (x1, y1, x2, y2)

    # 가장 긴 선분 그리기
    if longest_line is not None:
        x1, y1, x2, y2 = longest_line
        color = colors[idx % len(colors)]  # 색상이 넘치면 순환
        cv2.line(image, (x1, y1), (x2, y2), color, 3)  # 색상 적용하여 가장 긴 선분 그리기
        
        # 가장 긴 선분 옆에 숫자 1, 2, 3, 4 추가
        text_position = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)  # 선분의 중앙 위치
        cv2.putText(image, str(idx + 1), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 가장 긴 하얀색 선분 찾기
max_length_white = 0
best_line_white = None

for line in lines:
    for x1, y1, x2, y2 in line:
        length = get_length(x1, y1, x2, y2)
        if length > max_length_white:
            max_length_white = length
            best_line_white = (x1, y1, x2, y2)

# 가장 긴 하얀색 선분 그리기
if best_line_white is not None:
    x1, y1, x2, y2 = best_line_white
    cv2.circle(image, (x1, y1), 2, (0, 0, 255), -1)
    cv2.circle(image, (x2, y2), 2, (0, 0, 255), -1)
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 초록색 엣지 검출 결과 화면에 표시
cv2.imshow('Green Edges', edges_green)

# 결과 이미지 출력
cv2.imshow('Detected Lanes with Numbers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
