from easydict import EasyDict as edict


hp = edict()
hp.num_stations = 3
hp.max_gradients = 2
hp.time_step = 72
hp.output_time_step = 72
hp.inp_nume = 17
hp.inp_categ = 9
hp.num_layers = 2
hp.drop_out_rate = 0
hp.batch_size = 64
hp.embedding_dim = 5
hp.num_hidden = 128
hp.epoch = 60

ph = edict()
ph.start_date = '20180915'
ph.end_date = '20181219'
ph.name = 'chuzhou'

db = edict()
db.host = 'localhost'
db.port = 3306
db.user = 'root'
db.password = '95279527'
db.db = 'database'


def weather_encoder(weather):
    # convert the angle to the direction
    # angle: [0, 359], direction: [1, 8]
    weather_dict = {
        '晴': 0,
        '阴': 12,
        '多云': 8,
        '雾': 3,
        '浮尘': 3,
        '扬沙': 3,
        '雨': 4,
        '雪': 4,
        '雨夹雪': 4}
    return weather_dict[weather]


def wind_encoder(wind_direction):
    # convert wind direction
    wind_direction_dict = {
        '北风': 0,
        '东风': 2,
        '东北风': 1,
        '东南风 ': 3,
        '南风': 4,
        '西南风': 5,
        '西风': 6,
        '西北风': 7,
        '静风': 8
    }

    return wind_direction_dict[wind_direction]


def wind_speed_decoder(wind_speed):
    # convert wind speed
    wind_speed_dict = {
        '0': 0.1,
        '1': 1.0,
        '2': 2.5,
        '3': 4.0,
        '4': 6.7,
        '5': 9.4,
        '6': 11.5,
        '7': 15.5,
        '8': 19.0
    }
    return wind_speed_dict[wind_speed]
