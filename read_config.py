import sys
import logging
logger = logging.getLogger(__name__)
from configparser import ConfigParser
import dedalus.public as d3

class ConfigEval(ConfigParser, dict):

    def execute_defaults(self):
        vars = {}
        dflt = self.set_defaults(self.default, self.section)
        for key, val in dflt.items(dflt.section):
            if (key == 'dt'):
                key='timestep'
                exec("vars[\'{}\'] = {}".format(key, dflt.settings['dt']))
                continue
            try:
                exec("vars[key] = {}".format(val))

            except:
                exec("vars[\'{}\'] = \'{}\'".format(key, val))
            
        return vars


    def execute_locals(self):
        self.vars = self.execute_defaults()
        # logger.info(self.vars)
        # sys.exit()
        for key, val in self.items(self.section):
            if (key == 'dt'):
                key='timestep'
                exec("self.vars[\'{}\'] = {}".format(key, self.settings['dt']))
                continue
            elif (key == 'suffix'):
                self.vars['suffix'] = str(val)
                continue
            try:
                exec("self.vars[key] = {}".format(val))
            except:
                try:
                    self.vars[key] = str(val)
                except:
                    print(key)
                    sys.exit()
                    # self.vars[key] = self.settings[key]
            
        for key in self.aliases.keys():
            exec("self.vars[\'{}\'] = {}".format(key, self.aliases[key]))

        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        mainname = module.__file__
        logger.info('Running {} with the following parameters:'.format(mainname))
        logger.info(self.vars)
        return self.vars


    def forward_aliases(self):
        self.aliases = {}
        # self.aliases['Reynolds'] = self.parse('Re')
        # self.aliases['max_timestep'] = self.parse('dt')

        # self.settings['Schmidt'] = 1
        # self.settings['nu'] = 1 / self.parse('Re')
        # self.settings['D'] = self.settings['nu'] / self.settings['Schmidt']

    def __init__(self, filename, section='parameters', default='config.cfg'):
        super().__init__()
        self.optionxform = str
        self.settings = {}
        self.read(filename)
        for key, value in self.items(section):
            self.settings[str(key)] = self.parse(str(key))

        self.section = section
        self.default = default
        self.forward_aliases()


    def set_defaults(self, filename, section):
        if (filename == None):
            return
        else:
            return ConfigEval(filename, default=None)
        # for (key, value) in self.items(section):
            # self.settings[key] = self.parse(key)
            # self.default[key] = self.parse(key)

    def items(self, section):
        self.section = section
        return ConfigParser.items(self, section)

    def parse(self, arg, default=None):
        try:
            return eval(self.get(self.section, arg))
        except:
            if arg in self.settings.keys():
                return self.settings[arg]
            try:
                return str(self.get(self.section, arg))
            except:
                logger.info('Failed to read config property: {}'.format(arg))
                if (default == None):
                    logger.info('Default config property not supplied. terminating...')
                    sys.exit()
                else:
                    logger.info('Failed to read config property: {}'.format(arg))
                    logger.info('Default supplied: {}'.format(default))
                return default

    def SBI_dictionary(self, config, section='SBI'):
        self.read(config)
        dict = {}
        try:
            for key, value in self.items(section):
                dict[str(key)] = eval(str(value))
        except:
            logger.info('SBI section not found in config. proceeding with defaults')
        return dict


