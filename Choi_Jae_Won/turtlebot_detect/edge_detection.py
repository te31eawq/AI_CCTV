import cv2
import numpy as np

# 웹캠 열기
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://10.10.14.88:8080/?action=stream')
# HSV 범위 슬라이더 값 설정
def nothing(x):
    pass

# 윈도우 창 생성
cv2.namedWindow("Trackbars")

# 슬라이더 생성 (각각 Hue, Saturation, Value 범위)
cv2.createTrackbar('LH', 'Trackbars', 0, 255, nothing)  # Lower Hue
cv2.createTrackbar('UH', 'Trackbars', 0, 255, nothing)  # Upper Hue
cv2.createTrackbar('LS', 'Trackbars', 0, 255, nothing)  # Lower Saturation
cv2.createTrackbar('US', 'Trackbars', 0, 255, nothing)  # Upper Saturation
cv2.createTrackbar('LV', 'Trackbars', 0, 255, nothing)  # Lower Value (높은 명도)
cv2.createTrackbar('UV', 'Trackbars', 0, 255, nothing)  # Upper Value

# 초록색 범위 슬라이더 생성
cv2.createTrackbar('Lower Green H', 'Trackbars', 0, 255, nothing)  # 초록색의 Hue
cv2.createTrackbar('Upper Green H', 'Trackbars', 0, 255, nothing)  # 초록색의 Hue
cv2.createTrackbar('Lower Green S', 'Trackbars', 0, 255, nothing)  # 초록색의 Saturation
cv2.createTrackbar('Upper Green S', 'Trackbars', 0, 255, nothing)  # 초록색의 Saturation
cv2.createTrackbar('Lower Green V', 'Trackbars', 0, 255, nothing)  # 초록색의 Value
cv2.createTrackbar('Upper Green V', 'Trackbars', 0, 255, nothing)  # 초록색의 Value

# 창 크기 변경
cv2.resizeWindow("Trackbars", 800, 600)  # 창 크기를 800x600으로 설정

while True:
    # 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        break

    # HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 현재 슬라이더 값 읽기 (흰색 범위)
    l_h = cv2.getTrackbarPos('LH', 'Trackbars')
    u_h = cv2.getTrackbarPos('UH', 'Trackbars')
    l_s = cv2.getTrackbarPos('LS', 'Trackbars')
    u_s = cv2.getTrackbarPos('US', 'Trackbars')
    l_v = cv2.getTrackbarPos('LV', 'Trackbars')
    u_v = cv2.getTrackbarPos('UV', 'Trackbars')

    # 현재 슬라이더 값 읽기 (초록색 범위)
    l_green_h = cv2.getTrackbarPos('Lower Green H', 'Trackbars')
    u_green_h = cv2.getTrackbarPos('Upper Green H', 'Trackbars')
    l_green_s = cv2.getTrackbarPos('Lower Green S', 'Trackbars')
    u_green_s = cv2.getTrackbarPos('Upper Green S', 'Trackbars')
    l_green_v = cv2.getTrackbarPos('Lower Green V', 'Trackbars')
    u_green_v = cv2.getTrackbarPos('Upper Green V', 'Trackbars')

    # 흰색 범위 설정
    lower_white = np.array([l_h, l_s, l_v])
    upper_white = np.array([u_h, u_s, u_v])

    # 초록색 범위 설정
    lower_green = np.array([l_green_h, l_green_s, l_green_v])
    upper_green = np.array([u_green_h, u_green_s, u_green_v])

    # 흰색 마스크 생성
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # 초록색 마스크 생성
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 두 마스크 합치기
    mask = cv2.bitwise_or(mask_white, mask_green)

    # 마스크를 원본 이미지에 적용하여 결과 추출
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 결과 출력
    cv2.imshow('Filtered Frame', result)

    # 'q'를 눌러서 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        # 'q'를 눌러서 종료
    if cv2.waitKey(1) & 0xFF == ord('f'):
        cv2.imwrite('image.jpg', result)
# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
