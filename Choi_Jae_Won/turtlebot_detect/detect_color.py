import cv2
import numpy as np

# 마우스 클릭 시 호출될 함수
def get_color_info(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 클릭 시
        # 클릭한 위치의 BGR 색상 값
        bgr_color = img[y, x]
        # BGR을 HSV로 변환
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        h, s, v = hsv_color  # 색상, 채도, 명도 값 추출
        
        print(f"Clicked position: ({x}, {y})")
        print(f"BGR Color: {bgr_color}")
        print(f"HSV Color: {hsv_color}")
        print(f"Hue: {h}, Saturation: {s}, Value: {v}")

# 비디오 캡쳐 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

while True:
    success, img = cap.read()
    if not success:
        break  # 비디오 끝

    # 이미지를 화면에 표시 (먼저 화면에 띄운 후)
    cv2.imshow("Image", img)

    # 'cv2.setMouseCallback'를 사용하여 마우스 클릭 이벤트 처리
    cv2.setMouseCallback("Image", get_color_info)

    # 'q' 키를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
