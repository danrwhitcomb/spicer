import keras
from keras.layers.recurrent import Recurrent
from keras import backend as K

import numpy as np

'''
Prototype LSTM node
(untested)
'''
class LSTMNode:

    def __init__(self, val_shape, state_shape, output_shape, dropout):

        #inputs
        x = K.placeholder(shape=val_shape)
        h_1 = K.placeholder(shape=val_shape)
        C_1 = K.placeholder(shape=state_shape)

        #Forget gate
        comb = K.concatenate([h_1, x])
        Wf = K.random_normal_variable(shape=K.int_shape(comb),
                                            mean=0,
                                            scale=1,
                                            name='forget_W')
        bf = K.random_normal_variable(shape=K.int_shape(comb),
                                            mean=0.3,
                                            scale=1,
                                            name='forget_b')
        ft = self._sigmoid_node(x, h_1, Wf, bf)

        #input gate layer
        Wi = K.random_normal_variable(shape=K.int_shape(comb), mean=0, scale=1, name='i_W')
        bi = K.random_normal_variable(shape=K.int_shape(comb), mean=0, scale=1, name='i_b')
        it = self._sigmoid_node(x, h_1, Wi, b)

        #new state inclusion layer
        Wcl = K.random_normal_variable(shape=K.int_shape(comb), mean=0, scale=1, name='cl_W')
        bcl = K.random_normal_variable(shape=K.int_shape(comb), mean=0, scale=1, name='cl_b')
        CLt = K.tanh((Wc * comb) + bc)

        #New state
        self.Ct = (ft * C_1) + (it * CLt)

        #Output layer
        Wo = K.random_normal_variable(shape=K.int_shape(comb), mean=0, scale=1, name='o_W')
        bo = K.random_normal_variable(shape=K.int_shape(comb), mean=0, scale=1, name='o_b')
        ot = self._sigmoid_node(x, h_1, Wo, bo)
        self.ht = ot * K.tanh(Ct)

    def _sigmoid_node(self, x, h_1, W, b):
        combine = K.concatenate([h_1, x])
        return K.sigmoid((W * combine) + b)
