import cv2
import numpy as np
import math

def lane_detection(image):
    height, width = image.shape[:2]

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 색상 범위 설정
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])

    lower_blue = np.array([85, 50, 150])   
    upper_blue = np.array([105, 255, 255]) 

    # 마스크 생성
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    draw_image = image.copy()

    yellow_points = []
    white_points = []
    blue_points = []

    def process_contours(contours, points_list, color):
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) > 4:  # 오각형 이상의 다각형은 무시
                    continue
                elif len(approx) == 2:  # 선으로 인식했을 때
                    points = approx.reshape(2, 2)
                    points_list.append(points.tolist())
                    cv2.line(draw_image, tuple(points[0]), tuple(points[1]), color, 2)

                elif len(approx) == 3:  # 삼각형으로 인식했을 때
                    points = approx.reshape(3, 2)
                    distances = [
                        math.sqrt((points[i][0] - points[(i + 1) % 3][0]) ** 2 +
                                  (points[i][1] - points[(i + 1) % 3][1]) ** 2)
                        for i in range(3)
                    ]
                    min_index = distances.index(min(distances))
                    shortest_midpoint = ((points[min_index] + points[(min_index + 1) % 3]) // 2)
                    opposite_point = points[(min_index + 2) % 3]
                    points_list.append([shortest_midpoint.tolist(), opposite_point.tolist()])
                    cv2.line(draw_image, tuple(shortest_midpoint), tuple(opposite_point), color, 2)

                elif len(approx) == 4:  # 사각형으로 인식했을 때
                    points = approx.reshape(4, 2)
                    distances = [
                        math.sqrt((points[i][0] - points[(i + 1) % 4][0]) ** 2 +
                                  (points[i][1] - points[(i + 1) % 4][1]) ** 2)
                        for i in range(4)
                    ]
                    min_index = distances.index(min(distances))
                    midpoints = [
                        ((points[min_index] + points[(min_index + 1) % 4]) // 2),
                        ((points[(min_index + 2) % 4] + points[(min_index + 3) % 4]) // 2)
                    ]
                    points_list.append([midpoints[0].tolist(), midpoints[1].tolist()])
                    cv2.line(draw_image, tuple(midpoints[0]), tuple(midpoints[1]), color, 2)

    # 노란색 윤곽선 처리
    _, yellow_binary = cv2.threshold(yellow_mask, 127, 255, cv2.THRESH_BINARY)
    yellow_contours, _ = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    process_contours(yellow_contours, yellow_points, (0, 255,0))

    # 흰색 윤곽선 처리
    _, white_binary = cv2.threshold(white_mask, 127, 255, cv2.THRESH_BINARY)
    white_contours, _ = cv2.findContours(white_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    process_contours(white_contours, white_points, (255, 255, 255))

    # 파란색 윤곽선 처리
    _, blue_binary = cv2.threshold(blue_mask, 127, 255, cv2.THRESH_BINARY)
    blue_contours, _ = cv2.findContours(blue_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    process_contours(blue_contours, blue_points, (255, 0, 0))

    return draw_image, yellow_points, white_points, blue_points
