import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import Sort
import socket
import threading

# 비디오 캡쳐 및 첫 번째 프레임 가져오기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# YOLO 모델 로딩
model = YOLO("./best2.pt")

# 클래스 이름 설정 (차량만)
classNames = ["vehicle"]

# 비디오 프레임 크기 가져오기
success, img = cap.read()
if not success:
    print("Error: Couldn't read the first frame of the video")
    exit()
frame_height, frame_width = img.shape[:2]

# SORT 추적기 초기화
tracker = Sort(max_age=100, min_hits=2, iou_threshold=0.3)

# 차량 추적 정보 저장
tracked_vehicles = {}
id_counter = 1  # 차량 ID 생성용
id_map = {}

# 빨간색 상자 표시 여부 저장
red_box_flags = {}

# 픽셀당 실제 거리 (1 픽셀 = 몇 미터인지 설정)
scale = 0.05  # 1 픽셀 = 0.05m (실제 상황에 따라 조정)

# 평균 속도 추적 변수
last_avg_speed = 0
frame_buffer = {}  # 차량별 속도 저장 버퍼
buffer_size = 7  # 속도 계산에 사용할 프레임 수

# 네트워크 소켓 코드
BUF_SIZE = 100
NAME_SIZE = 20
ARR_CNT = 5

name = "CCTV"
msg = ""

SERVER_IP = "10.10.14.28"
SERVER_PORT = 5000

sock = None

# 사고 및 정상 상태 메시지 전송 여부
accident_sent = False
okay_sent = False

# 두 점 사이의 맨해튼 거리 계산 함수
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 차량 위치에 대한 부드럽게 처리하는 필터
def smooth_position(prev_pos, current_pos, alpha=0.5):
    smoothed_x = alpha * current_pos[0] + (1 - alpha) * prev_pos[0]
    smoothed_y = alpha * current_pos[1] + (1 - alpha) * prev_pos[1]
    return (int(smoothed_x), int(smoothed_y))

# 네트워크 소켓 코드
def send_msg(sock, message):
    global msg
    while True:
        msg = message
        
        if msg == "quit":
            if sock:
                sock.sendall(b'quit\n')
                sock.close()
            break
        
        if not msg.startswith('['):
            msg = f"[ALLMSG]{msg}"

        try:
            if sock:  # sock이 None이 아닌 경우에만 sendall 호출
                sock.sendall(msg.encode())
                break
            else:
                print("Error: socket is None.")
                break
        except:
            print("Connection lost. Exiting...")
            if sock:
                sock.close()
            break

def recv_msg(sock):
    while True:
        try:
            data = sock.recv(NAME_SIZE + BUF_SIZE)
            if not data:  # 서버가 연결을 끊으면 data가 None이거나 길이가 0이 됨
                print("Server disconnected. Exiting...")
                break
            print(data.decode())
        except:
            print("Connection lost. Exiting...")
            break

def socket_thread():
    global name, sock

    server_ip = SERVER_IP
    server_port = SERVER_PORT

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((server_ip, server_port))
        print(f"Connected to {server_ip}:{server_port}")

        login_msg = f"[{name}:PASSWD]"
        sock.sendall(login_msg.encode())

        message_to_send = "Server Connected\n"  # 예시 메시지
        send_thread = threading.Thread(target=send_msg, args=(sock, message_to_send))
        recv_thread = threading.Thread(target=recv_msg, args=(sock,))

        send_thread.start()
        recv_thread.start()

        send_thread.join()
        recv_thread.join()

    except Exception as e:
        print(f"Error: {e}")
        if sock:
            sock.close()
    finally:
        print("Connection closed.")

# 비디오 처리 루프
def video_thread():
    global last_avg_speed, id_counter, accident_sent, okay_sent
    while True:
        success, img = cap.read()
        if not success:
            break  # 비디오 끝

        # YOLO 모델을 통해 객체 감지
        results = model(img, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # 차량만 탐지
                if cls < len(classNames) and classNames[cls] == 'vehicle' and conf > 0.4:
                    currentarray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentarray))

        # SORT 추적기 업데이트
        tracked_objects = tracker.update(detections)
        accident_detected = False  # 사고 여부 체크 변수
        red_box_detected = False  # 빨간색 상자가 그려지는 차량 존재 여부

        for obj in tracked_objects:
            x1, y1, x2, y2, id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 차량 추적 정보 저장 및 갱신
            vehicle_id = id_map.get(id, id_counter)
            if id not in id_map:
                id_map[id] = vehicle_id
                id_counter += 1

            if id in tracked_vehicles:
                prev_centroid = tracked_vehicles[id]['centroid']
                
                # 위치 부드럽게 처리
                smoothed_centroid = smooth_position(prev_centroid, (cx, cy))
                cx, cy = smoothed_centroid

                distance = manhattan_distance(prev_centroid, (cx, cy))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30  # FPS 가져오기 (기본값: 30)
                time_interval = 1 / fps
                speed = (distance / time_interval) * scale * 3.6  # 픽셀 -> m/s -> km/h

                # 속도 계산
                if id not in frame_buffer:
                    frame_buffer[id] = []
                frame_buffer[id].append(speed)
                if len(frame_buffer[id]) > buffer_size:
                    frame_buffer[id].pop(0)

                avg_speed = np.mean(frame_buffer[id])
                tracked_vehicles[id]['speed'] = avg_speed

                # 빨간색 상자 표시 여부 결정 (속도가 1.8 km/h 이하인 경우 빨간색)
                if avg_speed <= 1.8:
                    red_box_flags[id] = True  # 속도가 1.8 이하라면 빨간색으로 표시
                    box_color = (0, 0, 255)  # 빨간색
                    accident_detected = True  # 빨간색 상자 표시된 차량이 있으면 사고 발생
                    red_box_detected = True
                else:
                    if id not in red_box_flags:  # 빨간색 상자가 그려지지 않은 차량은 파란색
                        red_box_flags[id] = False
                    # 빨간색으로 이미 표시된 차량은 계속 빨간색으로 유지
                    box_color = (0, 0, 255) if red_box_flags[id] else (255, 0, 0)

                # 속도 및 ID 표시
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                cvzone.putTextRect(img, f'ID: {vehicle_id} Speed: {avg_speed:.2f} km/h', 
                                   (x1, y1 - 10), scale=1, thickness=2, offset=1, colorT=(255, 255, 255))

                tracked_vehicles[id]['centroid'] = (cx, cy)
            else:
                tracked_vehicles[id] = {'centroid': (cx, cy), 'speed': 0}

        # 사고가 감지되면 서버에 메시지 전송 (단 한번만)
        if accident_detected and not accident_sent:
            message_to_send = "Accident\n"
            send_thread = threading.Thread(target=send_msg, args=(sock, message_to_send))
            send_thread.start()
            accident_sent = True  # 사고 메시지 전송 상태 업데이트
            okay_sent = False

        # 모든 차량에 대해 빨간색 상자가 그려지지 않으면 "It's Okay" 메시지 전송 (단 한번만)
        if not red_box_detected and not okay_sent:
            message_to_send = "It's Okay\n"
            send_thread = threading.Thread(target=send_msg, args=(sock, message_to_send))
            send_thread.start()
            okay_sent = True  # 정상 메시지 전송 상태 업데이트
            accident_sent = False  # 사고 상태 리셋 (다시 사고 메시지가 전송될 수 있도록)

        # 평균 속도 표시
        all_speeds = [v['speed'] for v in tracked_vehicles.values() if 'speed' in v]
        avg_speed = np.mean(all_speeds) if all_speeds else last_avg_speed
        last_avg_speed = avg_speed
        cvzone.putTextRect(img, f'Average Speed: {avg_speed:.2f} km/h', 
                           (20, 30), scale=1, thickness=2, offset=1, colorT=(0, 255, 0))

        # 이미지를 640x480 크기로 리사이즈
        img_resized = cv2.resize(img, (640, 480))

        # 화면 출력
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF == ord('q')
        if key:  # ESC 키 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 소켓과 비디오 처리 각각 다른 스레드에서 실행
    threading.Thread(target=socket_thread, daemon=True).start()
    video_thread()
