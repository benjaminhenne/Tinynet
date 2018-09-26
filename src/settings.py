from activation_functions import *

class Settings:

    def __init__(self):
        self.l1_regularize = True
        self.l1_regularizer_scale = 0.05

        self.af_inits = {'swish'    : lambda fan_in: 0.1, #swish
                         'relu'     : lambda fan_in: 2.0/fan_in[2] if len(fan_in) == 4 else 2.0/fan_in[1],  #relu
                         'elu'      : lambda fan_in: 0.1,   #elu
                         'tanh'     : lambda fan_in: fan_in[2]**(-1/2) if len(fan_in) == 4 else fan_in[1]**(-1/2),   #tanh
                         'identity' : lambda fan_in: 0.1   #identity
                        }
        self.af_set = [
                       [swish, 'swish'],
                       [tf.nn.relu, 'relu'],
                       [tf.nn.elu,'elu'],
                       [tf.nn.tanh, 'tanh'],
                       [identity_activation, 'identity']
                        ]
        self.network_structure = "ianntf"
        self.optimizer = "Adam"

        self.l2_lambda = 0.01
        self.l2_regularize = False

        #Dropout
        self.keep_prob = 0.5
