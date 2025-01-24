import socket
import threading

BUF_SIZE = 100
NAME_SIZE = 20
ARR_CNT = 5

name = "CJW_PY"
msg = ""

# 고정된 IP와 포트
SERVER_IP = "10.10.14.28"
SERVER_PORT = 5000

def send_msg(sock, message):
    global msg
    while True:
        msg = message
        
        if msg == "quit":
            sock.sendall(b'quit\n')
            sock.close()
            break
        
        if not msg.startswith('['):
            msg = f"[ALLMSG]{msg}"

        try:
            sock.sendall(msg.encode())
            break
        except:
            print("Connection lost. Exiting...")
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

def main():
    global name

    server_ip = SERVER_IP
    server_port = SERVER_PORT

    # 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((server_ip, server_port))
        print(f"Connected to {server_ip}:{server_port}")

        # 로그인 메시지 전송
        login_msg = f"[{name}:PASSWD]"
        sock.sendall(login_msg.encode())

        # 메시지를 미리 설정하여 보내기
        message_to_send = "[ALLMSG]Hello from CJW_PY!\n"  # 예시 메시지
        send_thread = threading.Thread(target=send_msg, args=(sock, message_to_send))
        recv_thread = threading.Thread(target=recv_msg, args=(sock,))

        send_thread.start()
        recv_thread.start()

        send_thread.join()
        recv_thread.join()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()
