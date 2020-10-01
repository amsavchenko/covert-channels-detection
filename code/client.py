from datetime import datetime
import errno
import socket
import sys
from time import sleep

from tqdm import tqdm
import yaml


config = yaml.safe_load(open('../config.yml'))
ip = config['server']['ip']
port = config['server']['port']

message_length = config['length']
encoding_type = config['encoding_type']
username_length = config['username_length']
bits_per_symbol = config['bits_per_symbol']


def convert_message_to_bytes(message):
    encoded_message = message.encode(encoding_type)
    message = ''
    for symbol in encoded_message:
        message += format(symbol, f'0{bits_per_symbol}b')
    return message


def convert_bytes_to_message(bin_message):
    message = ''
    for i in range(0, len(bin_message), bits_per_symbol):
        message += chr(int(bin_message[i:i + bits_per_symbol], 2))
    return message


class Client():
    def __init__(self, username, ip, port, delay_function=lambda x: 0.0):
        self.username = username

        self.delay_function = delay_function
        self.delay_counters = []
        self.inter_packet_arrival_times = []

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((ip, port))
        self.client_socket.setblocking(False)
        self.send_username(self.username)


    def send_message(self, delay_function, message, *args):
        message = message.encode(encoding_type)
        sleep(delay_function(message, *args))
        self.client_socket.send(message)


    def send_username(self, username):

        def control_username_length(username, length):
            if len(username) < length:
                username = username + (' ' * (length - len(username)))
            return username[:length]

        username = control_username_length(username, username_length)
        username = username.encode(encoding_type)
        self.client_socket.send(username)


    def send_messages(self, receiver, messages, covert_message='Test message', embed_every=3, print_progress=True):
        for i in tqdm(range(len(messages))):
            self.send_message(self.delay_function, messages[i])
            receiver.receive_message()


    def receive_message(self, verbose=False):
        while True:
            try:
                message = self.client_socket.recv(message_length)
                self.delay_counters.append(datetime.now())
                if verbose:
                    print(f'({self.username}) Received message: {message.decode(encoding_type)}')
                break

            except IOError as e:
                if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                    print(f'Reading error: {str(e)}')
                    sys.exit()
                continue
            except Exception as e:
                print(f'Reading error: {str(e)}')
                sys.exit()


    def calculate_inter_packet_delays(self):
        for i in range(1, len(self.delay_counters)):
            self.inter_packet_arrival_times.append((self.delay_counters[i] - self.delay_counters[i-1]).total_seconds())
        return self.inter_packet_arrival_times


    def extract_covert_message_from_inter_packet_delays(self, extraction_function):
        bit_message =  extraction_function(self.inter_packet_arrival_times)
        covert_message = convert_bytes_to_message(bit_message)
        return covert_message



class ClientWithCovertChannel(Client):
    def __init__(self, username, ip, port, delay_function, covert_channel_delay_function):
        Client.__init__(self, username, ip, port, delay_function)
        self.covert_channel_delay_function = covert_channel_delay_function

    def send_messages(self, receiver, messages, covert_message='Test message', embed_every=3, print_progress=True):
        covert_message = convert_message_to_bytes(covert_message)
        for i in tqdm(range(len(messages))):
            if i % embed_every == 0 and len(covert_message) > 0 and i > 0:
                self.send_message(self.covert_channel_delay_function, messages[i], int(covert_message[0]))
                covert_message = covert_message[1:]
            else:
                self.send_message(self.delay_function, messages[i])

            receiver.receive_message()




