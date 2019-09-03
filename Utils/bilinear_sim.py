import scipy.stats as stats

from keras import backend as K
from keras.engine.topology import Layer

class BilinearTensorLayer(Layer):
    def __init__(self, **kwargs):
        super(BilinearTensorLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        mean = 0.0
        std = 1.0
        # W : d*d
        self.input_dim = input_shape[0][1]
        initial_W_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(self.input_dim,self.input_dim))
        self.W = K.variable(initial_W_values)
        self.trainable_weights = [self.W]


    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('BilinearTensorLayer must be called on a list of tensors '
                          '(at least 2). Got: ' + str(inputs))
        batch_size = K.shape(inputs[0])[0]
        #tensor_1 = K.reshape(inputs[0], (1, self.input_dim))
        #tensor_2 = K.reshape(inputs[1], (1, self.input_dim))
        tensor_1 = inputs[0]
        tensor_2 = inputs[1]
        
        return K.reshape(K.sum(K.dot(tensor_1, self.W) * tensor_2, axis=-1), (batch_size, 1))
        #return K.reshape(K.dot(K.dot(K.transpose(tensor_1), self.W), tensor_2), (batch_size, self.output_dim))

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, 1)
    
    
class WBilinearScore(Layer):
    def __init__(self, **kwargs):
        super(WBilinearScore, self).__init__(**kwargs)


    def build(self, input_shape):
        mean = 0.0
        std = 1.0
        # W : d*d // [(None, 192), (None, 10)]
        self.U_input_dim = input_shape[1]-10
        self.W_input_dim = 10
        self.output_dim = 1

        self.W = self.add_weight(name='W_embeddings', 
                        shape=(self.W_input_dim,self.U_input_dim),
                        initializer='uniform',
                        trainable=True)
        self.U = self.add_weight(name='U_embeddings', 
                        shape=(self.U_input_dim,self.U_input_dim),
                        initializer='uniform',
                        trainable=True)
        super(WBilinearScore, self).build(input_shape)
        

    def call(self, inputs, mask=None):
        
        batch_size = K.shape(inputs)[0]
        tensor_1 = inputs[:,:-10]
        tensor_2 = inputs[:,-10:]
        pos = K.gradients(K.max(tensor_2, axis=-1), tensor_2)
        
        ## obtain W embeddings for rpos and rneg

        Wrpos = K.sum(K.reshape(pos, (batch_size, 10, 1))*self.W, axis=1)
        Wrneg = (K.sum(self.W, axis=0)-Wrpos)/(self.W_input_dim-1)
        
        ## obtain score value
        
        rUrpos = K.reshape(K.sum(K.dot(tensor_1, self.U) * Wrpos, axis=-1), (batch_size, 1))
        rUrneg = K.reshape(K.sum(K.dot(tensor_1, self.U) * Wrneg, axis=-1), (batch_size, 1))
        
        return K.reshape(K.tf.sigmoid(rUrpos - rUrneg), (batch_size, 1))
        

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 1)
    

class WBilinearTensorLayer(Layer):
    def __init__(self, **kwargs):
        super(WBilinearTensorLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        mean = 0.0
        std = 1.0
        # W : d*d // [(None, 192), (None, 10)]
        self.U_input_dim = input_shape[1]-10
        self.W_input_dim = 10
        self.output_dim = 1

        self.W = self.add_weight(name='W_embeddings', 
                        shape=(self.W_input_dim,self.U_input_dim),
                        initializer='uniform',
                        trainable=True)
        self.U = self.add_weight(name='U_embeddings', 
                        shape=(self.U_input_dim,self.U_input_dim),
                        initializer='uniform',
                        trainable=True)
#         self.bias = self.add_weight(name='bias',
#                                 shape=(self.output_dim,),
#                                 initializer='zeros',
#                                 trainable=True)
        super(WBilinearTensorLayer, self).build(input_shape)
        

    def call(self, inputs, mask=None):
        
        batch_size = K.shape(inputs)[0]
        tensor_1 = inputs[:,:-10]
        tensor_2 = inputs[:,-10:]

        pos = K.gradients(K.max(tensor_2, axis=-1), tensor_2)[0]
        neg = 1-pos
        
        ## obtain W embeddings for rpos and rneg

        Wrpos = K.sum(K.reshape(pos, (batch_size, 10, 1))*self.W, axis=1)
        Wrneg = K.reshape(neg, (batch_size, 10, 1))*self.W
        
        
        ## obtain score value
        rU = K.dot(tensor_1, self.U)
        rUrpos = K.reshape(K.sum(rU * Wrpos, axis=-1), (batch_size, 1))
        rUrneg = K.map_fn(lambda x : K.sum(x[1]*x[0], axis=-1), (rU,Wrneg), dtype=K.tf.float32)
        
        return K.reshape((rUrpos - rUrneg), (batch_size, 10)) #+self.bias?
        

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, 10)