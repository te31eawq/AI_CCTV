import cv2
import numpy as np

# 동영상 파일 경로
cap = cv2.VideoCapture(r'C:\Users\iot19\Documents\final_project\detectfile.mp4')  # 동영상 파일 경로

# 동영상 정보 (프레임 크기, FPS 등)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 동영상 FPS를 조정하여 재생 속도를 느리게 설정
target_fps = 800  # 목표 FPS
delay = int(1000 / target_fps)  # 1초를 목표 FPS로 나눈 지연 시간 (밀리초 단위)

# 선분 정보를 저장할 변수
best_line = None
prev_gray = None
prev_points = None
smooth_factor = 0.9  # 선분 위치를 스무스하게 만드는 인자 (0 ~ 1 범위)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 동영상 끝

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 매 프레임마다 선분을 검출하고 추적
    # 가우시안 블러링으로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    # 허프 변환으로 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 가장 긴 선분을 찾기 위한 변수 초기화
    max_length = 0

    # 모든 선분을 검사하여 가장 긴 선분을 찾기
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > max_length:
                    max_length = length
                    best_line = (x1, y1, x2, y2)

    # 첫 번째 검출된 선분의 두 끝점에 초록색 동그라미 그리기
    if best_line is not None:
        x1, y1, x2, y2 = best_line
        # 스무스한 선분 추적을 위해 이전 선분을 보간하여 조금씩 이동
        if prev_points is not None:
            # 두 점을 보간하여 스무스하게 이동
            x1 = int(prev_points[0][0] * smooth_factor + x1 * (1 - smooth_factor))
            y1 = int(prev_points[0][1] * smooth_factor + y1 * (1 - smooth_factor))
            x2 = int(prev_points[1][0] * smooth_factor + x2 * (1 - smooth_factor))
            y2 = int(prev_points[1][1] * smooth_factor + y2 * (1 - smooth_factor))

        # 선분 그리기
        cv2.circle(frame, (x1, y1), 2, (0, 255, 0), -1)  # 첫 번째 끝점에 동그라미
        cv2.circle(frame, (x2, y2), 2, (0, 255, 0), -1)  # 두 번째 끝점에 동그라미
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 직선 (두 점 연결)

    # Optical Flow를 이용한 점 추적
    if prev_points is not None:
        next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)

        # 추적된 점들에 대해 그리기
        for i, (new, old) in enumerate(zip(next_points, prev_points)):
            a, b = new.ravel()
            c, d = old.ravel()

            # 좌표를 정수형으로 변환하여 cv2.line()에 전달
            a, b, c, d = int(a), int(b), int(c), int(d)

            cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)  # 추적된 선 그리기
            cv2.circle(frame, (a, b), 2, (0, 255, 0), -1)   # 점을 그리기

        prev_points = next_points  # 추적된 점들을 새로운 prev_points로 설정

    # 추적할 점들을 초기화
    if best_line is not None:
        # best_line을 이용하여 추적할 점들 초기화 (2개의 점을 추적)
        prev_points = np.array([[best_line[0], best_line[1]], [best_line[2], best_line[3]]], dtype=np.float32)

    # 이전 프레임을 갱신
    prev_gray = gray
    # 실시간으로 결과를 화면에 표시
    cv2.imshow('Detected Lane', frame)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# 동영상 파일과 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
