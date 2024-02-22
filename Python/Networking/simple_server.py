import socket as skt
import threading
import pickle


HEADER = 64
PORT = 5050
SERVER = skt.gethostbyname(skt.gethostname())
ADDR = (SERVER , PORT)
FORMAT = 'utf-8'
DISCONECT_MESSAGE = "!DISCONNECT"

server = skt.socket(skt.AF_INET,skt.SOCK_STREAM)
server.bind(ADDR)

def handle_client(conn,addr):
    print(f"[NEW CONNECTION] {addr} connected")

    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONECT_MESSAGE:
                connected = False
        
            print(f"[{addr}] {msg}")
    conn.close()


def start():
    server.listen()
    print(f"[LISTENING] server is listnening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client
                                  , args=(conn ,addr))
        thread.start
        print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")


print("[STARTING] server is starting ...")
start()
