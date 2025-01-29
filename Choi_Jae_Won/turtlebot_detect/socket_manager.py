# socket_manager.py

import socket
import threading

BUF_SIZE = 100
NAME_SIZE = 20

class SocketManager:
    def __init__(self, server_ip, server_port, name="CCTV"):
        self.server_ip = server_ip
        self.server_port = server_port
        self.name = name
        self.sock = None
        self.msg = ""

    def send_msg(self, message):
        while True:
            self.msg = message
            if self.msg == "quit":
                if self.sock:
                    self.sock.sendall(b'quit\n')
                    self.sock.close()
                break
            if not self.msg.startswith('['):
                self.msg = f"[ALLMSG]{self.msg}"

            try:
                if self.sock:
                    self.sock.sendall(self.msg.encode())
                    break
                else:
                    print("Error: socket is None.")
                    break
            except:
                print("Connection lost. Exiting...")
                if self.sock:
                    self.sock.close()
                break

    def recv_msg(self):
        while True:
            try:
                data = self.sock.recv(NAME_SIZE + BUF_SIZE)
                if not data:  # 서버가 연결을 끊으면 data가 None이거나 길이가 0이 됨
                    print("Server disconnected. Exiting...")
                    break
                print(data.decode())
            except:
                print("Connection lost. Exiting...")
                break

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.sock.connect((self.server_ip, self.server_port))
            print(f"Connected to {self.server_ip}:{self.server_port}")

            login_msg = f"[{self.name}:PASSWD]"
            self.sock.sendall(login_msg.encode())

            message_to_send = "Server Connected\n"  # 예시 메시지
            send_thread = threading.Thread(target=self.send_msg, args=(message_to_send,))
            recv_thread = threading.Thread(target=self.recv_msg)

            send_thread.start()
            recv_thread.start()

            send_thread.join()
            recv_thread.join()

        except Exception as e:
            print(f"Error: {e}")
            if self.sock:
                self.sock.close()
        finally:
            print("Connection closed.")
