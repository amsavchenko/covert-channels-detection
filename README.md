# Обнаружение сетевых скрытых каналов по времени на основе алгоритмов машинного обучения

Скрытый канал по времени реализован следующим образом: 

1. написаны скрипты, эмулирующие чат между двумя пользователями
2. у отправителя есть возможность изменять время отправки сообщения (TCP-пакета) и с помощью этого передать сообщение по скрытому каналу
3. получатель измеряет длины межпакетных интервалов, с помощью заданных правил преобразует массив длин в двоичную последовательность и расшифровывает сообщение

## Подробнее о скрытом канале

Чат реализован с помощью библиотеки `socket`. 

- В `server.py` код для работы сервера, который может обслуживать произвольное число клиентов. Каждое новое пришедшее сообщение сервер рассылает всем подключенным к нему клиентам, кроме того, от которого сообщение пришло. 

- `client.py` содержит базовый класс `Client`, который позволяет отправлять и получать сообщения, и наследуемый от него класс `ClientWithCovertChannel`, который отличается тем, что при отправке сообщений с помощью временных задержек передает еще и скрытое сообщение. 

- `conversation.py` содержит код для эмуляции общения. В отсутствии скрытого канала задержка перед отправкой рассчитывается с учетом длины сообщения. Средняя скорость набора текста – 200 символов в минуту, поэтому формула выглядит следующим образом:

```python
# cps - char per second
average_cps = 200 / 60
delay_time = len(message) / average_cps
delay_time = delay_time * uniform(0.75, 1.25)
```

(`uniform(0.75, 1.25)` генерирует число из заданного диапазона. Левая и правая граница задаются через конфигурационный файл `config.yml`)

Во время работы скрытого канала задержка определяется следующим образом:

- если передается 0, то случайное число из отрезка [6, 12]
- если передается 1, то случайное число из отрезка [16, 22]

Эти границы так же задаются через `config.yml`. Если переданы все биты скрытого сообщения, но общение продолжается, задержка генерируется с учетом длины сообщения способом, описанным выше. 

С помощью параметра `embed_every` в `config.yml` можно задавать, как часто будут встраиваться биты скрытого сообщения. 

Посылаемые сообщения взяты из датасета с реальными сообщениями, [источник](https://www.kaggle.com/team-ai/spam-text-message-classification).

### Проверка работы скрытого канала

Для тестирования выбран простой случай:

- биты скрытого сообщения встраиваются в каждое 1-ое сообщение
- Скрытое сообщение: `'Qwerty12345'`,  его длина 11 символов, каждый кодируется 8 битами, следовательно необходимо 88 сообщений. 

В обоих случаях было отправлено 1000 сообщений, гистограммы длин межпакетных интервалов:

![overt_channel](https://github.com/amsavchenko/covert-channels-detection/blob/master/data/overt_traffic.png)

![covert_channel](https://github.com/amsavchenko/covert-channels-detection/blob/master/data/covert_traffic.png)

Если теперь оставить только первые 88 интервалов, то будет видна разница в распределениях:

![overt_channel](https://github.com/amsavchenko/covert-channels-detection/blob/master/data/overt_traffic_first88.png)

![overt_channel](https://github.com/amsavchenko/covert-channels-detection/blob/master/data/covert_traffic_first88.png)




