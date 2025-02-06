import socket
import threading

BUF_SIZE = 100
NAME_SIZE = 20

class SocketManager:
    def __init__(self, server_ip, server_port, name="CCTV1", callback=None):
        self.server_ip = server_ip
        self.server_port = server_port
        self.name = name
        self.sock = None
        self.msg = ""
        self.callback = callback  # 콜백을 초기화 시 받아들임

    def send_msg(self, message):
        """메시지를 보내는 메소드"""
        self.msg = message
        if self.msg == "quit":
            if self.sock:
                self.sock.sendall(b'quit\n')
                self.sock.close()
            return

        if not self.msg.startswith('['):
            self.msg = f"[ALLMSG]{self.msg}"

        try:
            if self.sock:
                self.sock.sendall(self.msg.encode())
            else:
                print("Error: socket is None.")
        except Exception as e:
            print(f"Connection lost. Exiting... {e}")
            if self.sock:
                self.sock.close()

    def recv_msg(self):
        """서버에서 메시지를 받는 메소드"""
        try:
            while True:
                data = self.sock.recv(BUF_SIZE)
                if not data:  # 서버가 연결을 끊으면 data가 None이거나 길이가 0이 됨
                    print("Server disconnected.")
                    break
                message = data.decode()
                print(f"Received message: {message}")
                if self.callback:
                    self.callback(message)  # 메시지를 처리하는 콜백 함수 호출
        except Exception as e:
            print(f"Error while receiving message: {e}")
        finally:
            self.sock.close()

    def connect(self):
        """서버와 연결하는 메소드"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.sock.connect((self.server_ip, self.server_port))
            print(f"Connected to {self.server_ip}:{self.server_port}")

            login_msg = f"[{self.name}:PASSWD]"
            self.sock.sendall(login_msg.encode())

            message_to_send = "Server Connected\n"
            send_thread = threading.Thread(target=self.send_msg, args=(message_to_send,))
            recv_thread = threading.Thread(target=self.recv_msg)

            send_thread.start()
            recv_thread.start()

            send_thread.join()  # send_msg 스레드가 종료될 때까지 기다림
            recv_thread.join()  # recv_msg 스레드가 종료될 때까지 기다림

        except Exception as e:
            print(f"Error: {e}")
            if self.sock:
                self.sock.close()
        finally:
            print("Connection closed.")

    def set_callback(self, callback):
        """외부에서 콜백을 설정할 수 있도록 함"""
        self.callback = callback
