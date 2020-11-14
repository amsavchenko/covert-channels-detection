import select
import socket

import yaml


config = yaml.safe_load(open('../config.yml'))
ip = config['server']['ip']
port = config['server']['port']
message_length = config['length']
encoding_type = config['encoding_type']
username_length = config['username_length']


def receive_message(client_socket, message_length=message_length):
        try:
            message = client_socket.recv(message_length)
            if message == b'':
                return False
            return message
        except:
            return False


def shrink_username_length(name):
    idx = name.find(' ')
    if idx < 1:
        return name
    return name[:idx]


class Server:
    def __init__(self, ip, port):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((ip, port))
        self.server_socket.listen()

        self.stop_working = False
        self.sockets_list = [self.server_socket]
        self.client_names = {}


    def running(self):
        while True:
            read_sockets, _, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)
            # print(len(self.sockets_list))
            for notified_socket in read_sockets:
                # means that new client wants to connect to server
                if notified_socket == self.server_socket:
                    # apply connection
                    client_socket, client_address = self.server_socket.accept()

                    # receive client's username
                    username = receive_message(client_socket, username_length)
                    if username is False:
                        continue
                    username = shrink_username_length(username.decode(encoding_type))
                    self.sockets_list.append(client_socket)
                    self.client_names[client_socket] = username
                    print(f'Accepted connection from {username}')
                # message to already connected client
                else:
                    message = receive_message(notified_socket)
                    if message is False:
                        print(f'Closed connection from {self.client_names[notified_socket]}')
                        self.sockets_list.remove(notified_socket)
                        # if len(self.sockets_list) == 1:
                            # self.stop_working = True
                        continue

                    # share message
                    for client_socket in self.sockets_list:
                        if client_socket != notified_socket and client_socket != self.server_socket:
                            client_socket.sendall(message)

            for notified_socket in exception_sockets:
                self.sockets_list.remove(notified_socket)

            if self.stop_working:
                break


if __name__ == '__main__':
    server = Server(ip, port)
    server.running()