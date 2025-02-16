# 목차
### 1. [차선 분리](#차선-감지) | 2025.01.23
### 2. [사고감지](#deepsort) | 2025.01.24
### 3. [터틀봇 사고감지](#turtlebot3_deepsort_client_2) | 2025.01.27


---


### 차선 감지

final_project/lane_detect_addition_3.py를 실행시


<p align = center>
<img src = "./images/lane.png" width = 400>
</p>

하얀선은 차선으로 <span style="color: red; font-weight: bold;">빨간색으로 표시</span>

초록색 선은 각 선마다 번호를 부여

<p align = center>
<img src = "./images/lane_detect_fail1.png" width = 600>
</p>


---

### Deepsort

<p align = center>
<img src = "./images/detect_normal.png", width = 500>

사고난 차량 감지 시 해당 차량의 색을 빨간색으로 표시

이후에 정상적인 속도여도 지속적으로 빨간색으로 표시


<p align = center>
<img src = "./images/detect_red.png", width = 500>
</p>

---

### turtlebot3_deepsort_client_2

서버에 접속 (Client : CCTV)

터틀봇에 대해 Deepsort 처리를 하여 각 객체마다의 속도를 계산한 후 1.8 이하의 속도로 내려가면
그 이후의 속도에 무관하게 카메라에서는 빨간색 상자로 지속적으로 표시

화면에 빨간색 상자로 처리되는 객체가 존재하면 서버로 "Accident" 라는 문구를 전송

<p align = center>
<img src = "./images/turtlebot_accident.png" width = 500>
</p>

후에 빨간색 상자로 처리되는 객체가 하나도 존재하지 않을 시 "Normal"이라는 문구를 보냄

<p align = center>
<img src = "./images/turtlebot_normal.png", width = 500>
</p>

---

### turtlebot3_deepsort_lane_client

<p align = center>
<img src = "./images/deepsort_lane_turtlebot.png", width = 500>
</p>

### turtlebot3_final

사고 발생 시 Accident@(1차선사고여부(0,1))@(2차선사고여부(0,1)) 메세지를 서버로 전송

※ 현재 사고난 차량의 차선 정보가 변경 시 
```
Accident@0@1
Accident@1@0
Accident@1@1
```

## 2025-02-03 update

### edge_detection.py

색 영역 추출용


### turtlebot_final_speed.py
```python
lane_detection2.py 사용 # opencv 차선 추출
socket_manager.py 사용 # Socket TCP/IP 서버 통신용
```

- 현재 차선을 지날 때 마다 차량마다 속도를 측정
- 5초이내에 초록색 차선이 업데이트 되지 않는다면 그 구간에서 사고가 났다고 판단
- **Accident@구간번호**를 서버로 보내도록 설정