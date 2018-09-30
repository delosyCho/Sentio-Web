import socket

HOST = ''  # 호스트를 지정하지 않으면 가능한 모든 인터페이스를 의미한다.
PORT = 9797  # 포트지정
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)  # 접속이 있을때까지 기다림
conn, addr = s.accept()  # 접속 승인

print('Connected by', addr)

while True:
    print()
    print('---------')

    data = conn.recv(1024)
    if not data: break
    data = str(data, encoding='utf-8')

    print(data)
    conn.send("ok\n".encode(encoding='utf-8'))  # 받은 데이터를 그대로 클라이언트에 전송

    #break

conn.close()
print("Closed")