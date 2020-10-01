import pickle
from random import uniform

import numpy as np
import pandas as pd
import yaml

from client import Client, ClientWithCovertChannel


config = yaml.safe_load(open('../config.yml'))
ip = config['server']['ip']
port = config['server']['port']
bits_per_symbol = config['bits_per_symbol']

covert = config['covert']
covert_message = config['covert_message']
embed_every = config['embed_every']
# Data from here: https://www.kaggle.com/team-ai/spam-text-message-classification
path_to_messages = config['path_to_messages']
number_of_messages = config['number_of_messages']
longest_message_length = config['longest_message_length']

if config['save']:
    if covert:
        path_to_save_file = config['path_to_save_file']['covert'][embed_every]
    else:
        path_to_save_file = config['path_to_save_file']['overt']

# cps - char per second
average_cps = 200 / 60


def load_messages():
    df = pd.read_csv(path_to_messages)
    real_messages = df.loc[df['Category'] == 'ham', 'Message']
    return real_messages[real_messages.apply(lambda x: len(x) < longest_message_length)]


def normal_delay(message):
    normal_delay_time = len(message) / average_cps
    return normal_delay_time * uniform(*config['normal_delay'].values())


def ip_ctc_delay(message, covert_bit):
    return uniform(*config['ip_ctc'][covert_bit].values())


def extract_from_ip_ctc(inter_packet_delays, default_value='0'):
    bit_message = ''
    for i in range(embed_every - 1, len(inter_packet_delays), embed_every):
        delay = inter_packet_delays[i]
        if delay >= config['ip_ctc'][0]['left'] and delay <= config['ip_ctc'][0]['right']:
            bit_message += '0'
        elif delay >= config['ip_ctc'][1]['left'] and delay <= config['ip_ctc'][1]['right']:
            bit_message += '1'
        else:
            bit_message += default_value
    return bit_message


if __name__ == '__main__':

    all_messages = load_messages()

    if covert:
        client1 = ClientWithCovertChannel('Client_1', ip, port, normal_delay, ip_ctc_delay)
    else:
        client1 = Client('Client_1', ip, port, normal_delay)

    client2 = Client('Client_2', ip, port, normal_delay)

    indices = np.random.randint(0, all_messages.shape[0], size=number_of_messages)
    messages = all_messages.iloc[indices].to_list()

    client1.send_messages(client2, messages, covert_message, embed_every)

    inter_packet_arrival_times = client2.calculate_inter_packet_delays()

    if config['save']:
        with open(path_to_save_file, 'wb') as file:
            pickle.dump(inter_packet_arrival_times, file)

    if covert:
        decoded_message = client2.extract_covert_message_from_inter_packet_delays(extract_from_ip_ctc)
        print(decoded_message)