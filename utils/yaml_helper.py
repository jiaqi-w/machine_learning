import yaml

class Yaml_Helper():
    def __init__(self, config_fname):
        self.config = self.load_config(config_fname) or {}

    @staticmethod
    def load_config(config_fname):
        """
        Loads the configuration file config_file given by command line or
        config.DEFAULT_CONFIG when none is specified. the default values
        are used for any configuration value that is not specified in config_file.
        :param args:
        :return: dictionary of config values
        """
        with open(config_fname) as config_file:
            conf = yaml.load(config_file)
        return conf

    def add_config(self, config_fname, should_override=False):
        """
        Please note that this function will always override the key values.
        :param config_fname:
        :return:
        """
        default = self.load_config(config_fname)
        for key in default:
            if key not in self.config or should_override and key in self.config:
                self.config[key] = default[key]
