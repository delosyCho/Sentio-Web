import Command_Processor
import socket

HOST = ''  # 호스트를 지정하지 않으면 가능한 모든 인터페이스를 의미한다.
PORT = 9797  # 포트지정
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)  # 접속이 있을때까지 기다림
conn, addr = s.accept()  # 접속 승인

print('Connected by', addr)
conn.send(('하이요' + '\n').encode(encoding='utf-8'))

processor = Command_Processor.Command_Processor()

command, result = processor.process('이동 팔')
command, result = processor.process('설정 게시판 신')

print('ready')
while True:
    data = conn.recv(1024)
    if not data: break
    data = str(data, encoding='utf-8')
    print(data)
    command, result = processor.process(data)
    print(command, result)
    if command == 1:
        result = str(result).replace('\n', ' ')
        conn.send((result + '\n').encode(encoding='utf-8'))