class Config(object):
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            if isinstance(val, dict):
                val = Config(val)
            self.__dict__.update({key: val})

    def __repr__(self):
        return str(self.__dict__)


if __name__ == "__main__":
    config_dict = {'gamma': 0.99, 'lr': {'policy': 0.0001, 'value': 0.0002}}
    config = Config(config_dict)
    print(config)
