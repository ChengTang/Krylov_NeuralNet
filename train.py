from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.client import timeline

import numpy as np
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from IPython.display import clear_output
import scipy
from scipy import optimize
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import pickle

from abc import ABC, abstractmethod

import pickle

import utils
from utils import experiment_abs, CatVariable
import importlib

importlib.reload(utils)

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes


from tensorflow.contrib.solvers.python.ops import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops

import collections
 ####################################################################

activation_functions = {'tanh':tf.nn.tanh,'softplus':tf.nn.softplus}

MODEL_ACTIVATION = 'tanh'
act = activation_functions[MODEL_ACTIVATION]


EXPNAME = 'EXP_KRYLOV'

DIRECTION_SEED = 0


REGULARIZATION_PARAM = 0#1e-10
NDATA = 1024 # number of data points.  324(over-param) or 1024(under-param)
DIM_Y = 1 # output dimension
DIM_IN = 2# input dimension

RESNET_NN = False # model to be learned;
DIM_HIDDEN = 128

def total_param(DIM_HIDDEN):
    return 2 * DIM_HIDDEN + DIM_HIDDEN + DIM_HIDDEN + 1

print('number of parameters = '+ str(total_param(DIM_HIDDEN)))

NOISE_LEVEL = 3

LEVEL_OF_NONLINEARITY = 3

# skip phase 1 if False
TWO_PHASE = True

# phase 1
TRAIN_EPOCHS_PHASE1 = 5000
DISPLAY_STEP_PHASE1 = 1000 # checkpoint step
MODEL_SAVE_NAME_PHASE1 = EXPNAME

LOGSTEP_PHASE1 = 100


# Phase 2
TRAIN_EPOCHS_PHASE2 = 10
DISPLAY_STEP_PHASE2 = 1
MODEL_SAVE_NAME_PHASE2 = EXPNAME 
SAMPLING_REGION = 2
SAMPLING_POINTS = 100

LOGSTEP_PHASE2 = 1


# Lanczos iteration
K = 128

THRESHOLD = 1e-2

GOLDENRATIO = 1.618033988749894848

def high_d_data_NN( m = NDATA, d = 2, d_hidden = 10, dim_y = 1):

    np.random.seed(0)
    X = np.random.uniform(-1, 1, size = (m,d))
    W1_true = np.random.uniform(-LEVEL_OF_NONLINEARITY, LEVEL_OF_NONLINEARITY, size = (d, int(d_hidden)))
    W2_true = np.random.uniform(-LEVEL_OF_NONLINEARITY, LEVEL_OF_NONLINEARITY, size = (int(d_hidden), dim_y))
    b1 = np.random.uniform(-1, 1, int(d_hidden))
    b2 = np.random.uniform(-1, 1, int(dim_y))

    
    x = np.cos (np.pi * np.arange(int(np.sqrt(m))) / (int(np.sqrt(m)) - 1))
    y = np.cos (np.pi * np.arange(int(np.sqrt(m))) / (int(np.sqrt(m)) - 1))
    x, y = np.meshgrid(x,y)
    
    X = np.vstack((x.flatten(), y.flatten())).T
    
    non_linear = np.dot(np.tanh(np.dot(X,W1_true) + b1), W2_true) + b2

    data = non_linear

    V = np.hstack([ W1_true.flatten(), b1.flatten(), W2_true.flatten(), b2.flatten()])


    np.random.seed(DIRECTION_SEED)
     
    noise = NOISE_LEVEL * np.random.normal(size = V.shape)

    XX, YY = np.mgrid[-1:1:100j, -1:1:100j]
    input_ = np.vstack((XX.flatten(), YY.flatten())).T
    non_linear = (np.dot(np.tanh(np.dot(input_,W1_true) + b1), W2_true) + b2).reshape(XX.shape)

    #fig,ax = plt.subplots(figsize = (7,7))
    #p = ax.contourf(XX, YY, non_linear, levels = 20)
    #fig.colorbar(p)
    #plt.show()

    
    return X, data, V, noise


# Hessian vector product 
def get_Hv_op(cost, params, vec):
    """ 
    Implements a Hessian vector product estimator Hv op defined as the 
    matrix multiplication of the Hessian matrix H with the vector v.

    Args:      
        v: Vector to multiply with Hessian (tensor)

    Returns:
        Hv_op: Hessian vector product op (tensor)
    """
    cost_gradient = utils.flatten(tf.gradients(cost, 
                                              params))
    vprod = tf.math.multiply(cost_gradient, 
                             tf.stop_gradient(vec))
    Hv_op = utils.flatten(tf.gradients(vprod, 
                                       params))

    return Hv_op

# return lanczos ops
def get_lanczos_ops(k,
           cost,
           params,
           q_holder,
           orthogonalize = True,
           name = 'lanczos'):



    def tarray(size, dtype, name):
        return tensor_array_ops.TensorArray(
            dtype=dtype, 
            size=size, 
            tensor_array_name=name, 
            clear_after_read = False)


    # tensor array stores vectors row-major
    def read_colvec(qholder, i):
        return array_ops.expand_dims(qholder[:,i], -1)


    
    def write_colvec(q_holder, colvec, i):
        updates = array_ops.expand_dims(colvec, -1)
        q_holder_shape = q_holder.get_shape()
        q_holder = tf.concat( [ q_holder[ :, : i ], updates, q_holder[ :, i + 1 : ] ], axis = 1 )
        q_holder.set_shape( q_holder_shape )
        return q_holder


    lanczos_state = collections.namedtuple("LanczosState",
                                            ["q", "alpha", "beta"])

    def update_state(old, i, q, alpha, beta):

        return lanczos_state(
            write_colvec(old.q, q, i+1),
            old.alpha.write(i, alpha),
            old.beta.write(i, beta))

    def orthogonalize_once(i, basis, v):

        Qt = tf.transpose(basis)[:i+1] # shape = k x n #array_ops.matrix_transpose(basis.stack() )
        # math_ops.matmul(Qt, math_ops.matmul(Qt, v)): shape = [n x k] x [k x n] x [n x 1]
        
        v-= math_ops.matmul(Qt, math_ops.matmul(Qt, v), adjoint_a = True)
        
        return v, util.l2norm(v)
        
    def orthogonalize_(i, basis, v):
        v_norm = util.l2norm(v)
        v_new, v_new_norm = orthogonalize_once(i, basis, v)
        # If the norm decreases more than 1/sqrt(2), run a second
        # round of MGS. See proof in:
        #   B. N. Parlett, ``The Symmetric Eigenvalue Problem'',
        #   Prentice-Hall, Englewood Cliffs, NJ, 1980. pp. 105-109
        return control_flow_ops.cond(v_new_norm < 0.7071 * v_norm,
                                     lambda: orthogonalize_once(i, basis, v),
                                     lambda: (v_new, v_new_norm))

    def stopping_criterion(i, _):
        # TODO(rmlarsen): Stop if an invariant subspace is detected.
        return i < k+1

    def lanczos_step(i, ls):
        q = read_colvec(ls.q, i)
        z = get_Hv_op(cost, params, array_ops.squeeze(q))

        z = array_ops.expand_dims(z,-1)
        alpha = array_ops.squeeze(math_ops.matmul(q, z, adjoint_a = True))

        if(orthogonalize):
            z, beta = orthogonalize_(i, ls.q, z)
        else:
            q_old = read_colvec(ls.q, i-1)
            beta_old =  read_colvec(ls.beta, i-1)
            z = z - alpha * q + beta_old * q_old 
            beta = util.l2norm(z)
            
        z = array_ops.squeeze(z)
        return i+1, update_state(ls, i, math_ops.divide(z,beta), alpha, beta)

    with ops.name_scope(name):
        dtype = tf.float64
        num_of_params = int(params.shape[0])
        
        np.random.seed(DIRECTION_SEED)
        
        starting_vector_np = np.random.rand(num_of_params)
        starting_vector = tf.constant(starting_vector_np, dtype = dtype)
        
        q1 = (starting_vector)/util.l2norm(starting_vector)
        q0 = array_ops.zeros([num_of_params], dtype = dtype)
        
        q_holder = write_colvec(q_holder, q0, 0)
        q_holder = write_colvec(q_holder, q1, 1)
        
        
        ls = lanczos_state(q = q_holder,
                          alpha = tarray(k+1, dtype, "alpha"),
                          beta = tarray(k+1, dtype, "beta").write(tf.constant(0), 0))
        
        i = constant_op.constant(1, dtype = dtypes.int32)
        
        _, ls = control_flow_ops.while_loop(stopping_criterion, lanczos_step, [i,ls])
        return lanczos_state(ls.q[:,1:-1], 
                             ls.alpha.stack()[1:],
                             ls.beta.stack()[1:-1])
        
# build tridiagonal matrix
def tridiagonal(diag, sub, sup):
    n = tf.shape(diag)[0]
    r = tf.range(n)
    ii = tf.concat([r, r[1:], r[:-1]], axis=0)
    jj = tf.concat([r, r[:-1], r[1:]], axis=0)
    idx = tf.stack([ii, jj], axis=1)
    values = tf.concat([diag, sub, sup], axis=0)
    return tf.scatter_nd(idx, values, [n, n])

def line_search(cost, V, dx, stepsize_max, ls_threshold, a0, b0, a1, b1, fa1, fb1):
    '''self.compute_cost, self.V, dx, 10, 1, 1e-8'''

    
    def update_fa1_fb1_true(a0, a1, b1, b0):
        a1_ = (2-GOLDENRATIO) * b1 + (GOLDENRATIO-1) * a0
        return a0, a1_, a1, b1, 1#tf.constant(1)

    def update_fa1_fb1_false(a0, a1, b1, b0):
        b1_ = (GOLDENRATIO-1) * b0 + (2-GOLDENRATIO) * a1
        return a1, b1, b1_, b0, 0#tf.constant(0)


    def line_search_step(i, a0, a1, b1, b0, fa1, fb1, changed, ls_threshold):



        def f0(a0, a1, b1, b0, fa1, fb1): return fb1, cost(V + dx * b1)
        def f1(a0, a1, b1, b0, fa1, fb1): return cost(V + dx * a1), fa1

        fa1, fb1 = tf.cond( tf.equal(changed, 0),
                            true_fn  = lambda: f0(a0, a1, b1, b0, fa1, fb1), 
                            false_fn =  lambda: f1(a0, a1, b1, b0, fa1, fb1))


        a0, a1, b1, b0, changed = tf.cond(fa1 < fb1,
                                          lambda: update_fa1_fb1_true(a0, a1, b1, b0),
                                          lambda: update_fa1_fb1_false(a0, a1, b1, b0))

        return i+1, a0, a1, b1, b0, fa1, fb1, changed, ls_threshold

    def stopping_criterion(i, a0, a1, b1, b0, fa1, fb1, changed, ls_threshold):
        # just check the threshold
        return b0 - a0 > ls_threshold

    a1 = (2-GOLDENRATIO) * b0 + (GOLDENRATIO-1) * a0
    b1 = (2-GOLDENRATIO) * a0 + (GOLDENRATIO-1) * b0

    fa1 = cost(V + dx * a1)
    fb1 = cost(V + dx * b1)

    i = 0 #tf.constant(0, dtype = dtypes.int32)
    
    a0, a1, b1, b0, changed = tf.cond(fa1 < fb1,
                                      lambda: update_fa1_fb1_true(a0, a1, b1, b0),
                                      lambda: update_fa1_fb1_false(a0, a1, b1, b0))


    
    return control_flow_ops.while_loop(stopping_criterion,
                               line_search_step,
                               [i, a0, a1, b1, b0, fa1, fb1, changed, ls_threshold])


class experiment_phase1(experiment_abs):
    
    def __init__(self, dim, 
                 hidden_dim, 
                 Ndata,
                 dim_y, 
                 train_epochs, 
                 display_step,
                 regularization_coeff,
                 RESNET,
                 name,
                 sampling_region = 0,
                 sampling_points = 0,
                 generate_data = high_d_data_NN):
    
        super(experiment_phase1, self).__init__(dim = dim, 
                 hidden_dim = hidden_dim, 
                 Ndata = Ndata ,
                 dim_y = dim_y , 
                 train_epochs = train_epochs, 
                 display_step = display_step ,
                 regularization_coeff = regularization_coeff,
                 generate_data = generate_data)
        
        
        #self.X = self.train_X
        #self.Y = self.train_Y
        
        shapes_RESNET = [(self.in_dim, self.out_dim),  # W
                       (1, self.out_dim), #b
                       (self.in_dim, self.hidden_dim), # W1
                       (1, self.hidden_dim), # b1
                       (self.hidden_dim, self.hidden_dim), # W2
                       (1, self.out_dim)] # b2
        
        shapes_NN = [(self.in_dim, self.hidden_dim), # W1
                       (1, self.hidden_dim), # b1
                       (self.hidden_dim, self.out_dim), # W2
                       (1, self.out_dim)] # b2
        
        self.RESNET = RESNET
        self.name = name
        self.sampling_region = sampling_region 
        self.sampling_points = sampling_points
        

        
        if(self.RESNET):
            self.shapes = shapes_RESNET
        else:
            self.shapes = shapes_NN
            
        def get_V_indexes(shapes):
            self.V_indexes = []
            s = 0
            for (z0, z1) in shapes:
                self.V_indexes.append((s, s+(z0*z1)))
                s += z0*z1
        get_V_indexes(self.shapes)

    def break_super_vector(self):
        # Random initialization
        if(self.RESNET):
            self.V, (self.W, self.b, self.W1, self.b1, self.W2, self.b2) = CatVariable(self.shapes)
            self.weights = [self.W, self.W1, self.W2] 
        else:   
            self.V, (self.W1, self.b1, self.W2, self.b2) = CatVariable(self.shapes)
            self.weights = [self.W1, self.W2]  
        # s_tweaked needed for optimization step.
    
    def set_threshold_holder(self):
        self.threshold = tf.Variable(tf.zeros(shape = (), dtype = tf.float64), name = "threshold_holder")
        
    def inference(self):

        # define inference
        
        nonlinear_part = (act(tf.linalg.matmul(self.inputs, self.W1) + self.b1))
        nonlinear_part = (tf.linalg.matmul(nonlinear_part, self.W2) + self.b2) ## zeros
        
        if(self.RESNET):
            linear_part = tf.linalg.matmul(self.inputs, self.W + self.b)
            self.pred = linear_part + nonlinear_part
        else:
            self.pred = nonlinear_part
            
    def two_phase_optimization(self, split = 0.90):
        self.two_phase_training = True
        self.second_phase_start = int(split * self.training_epochs)
    
    
    def operations(self):
        super(experiment_phase1, self).operations()

        ## training 
        self.set_threshold_holder()
        
        '''
        # newton's direction
        self.newton_ops = NewtonsMethod_avoid_saddle_points_modified_direction(self.cost, 
                                                                       self.V, 
                                                                       0)
        '''
    def train(self, k):


        with tf.Session() as sess:
            
            sess.run(self.init)

            
            # Fit all training data
            feed_dict_ = {self.X: self.train_X, self.Y: self.train_Y}

            self.Hessian = None
            self.gradient = sess.run(self.dc_dw_autodiff, feed_dict = feed_dict_)[0]
            self.V_val0 = sess.run(self.V, feed_dict = feed_dict_)

            ## store initial statistics
            c = sess.run(self.cost, feed_dict = feed_dict_)
            self.cost_hist.append(c)
            l = sess.run(self.loss, feed_dict = feed_dict_)
            self.loss_hist.append(l)            
            self.gradient_autodiff_norm = sess.run(self.dc_dw_autodiff_norm, feed_dict = feed_dict_)
            self.dc_dw_autodiff_norm_hist.append(self.gradient_autodiff_norm)
            

                
            
            for epoch in range(self.training_epochs):

                self.optimize('phase1_optimizer', sess, feed_dict_)
                if (epoch+1) % LOGSTEP_PHASE1 == 0:
                    c = sess.run(self.cost, feed_dict = feed_dict_)
                    self.cost_hist.append(c)

                    l = sess.run(self.loss, feed_dict = feed_dict_)
                    self.loss_hist.append(l)

                    self.gradient_autodiff_norm = sess.run(self.dc_dw_autodiff_norm, feed_dict = feed_dict_)
                    # store values
                    self.dc_dw_autodiff_norm_hist.append(self.gradient_autodiff_norm)
                    self.V_val = (sess.run(self.V, feed_dict = feed_dict_))

                    self.saver()

class experiment_phase2(experiment_abs):
    
    def __init__(self, dim, 
                 hidden_dim, 
                 Ndata,
                 dim_y, 
                 train_epochs, 
                 display_step,
                 regularization_coeff,
                 RESNET,
                 name,
                 sampling_region = 0,
                 sampling_points = 0,
                 generate_data = high_d_data_NN):
    
        super(experiment_phase2, self).__init__(dim = dim, 
                 hidden_dim = hidden_dim, 
                 Ndata = Ndata ,
                 dim_y = dim_y , 
                 train_epochs = train_epochs, 
                 display_step = display_step ,
                 regularization_coeff = regularization_coeff,
                 generate_data = generate_data)
        
        #self.X = self.train_X
        #self.Y = self.train_Y
        
        

        shapes_RESNET = [(self.in_dim, self.out_dim),  # W
                       (1, self.out_dim), #b
                       (self.in_dim, self.hidden_dim), # W1
                       (1, self.hidden_dim), # b1
                       (self.hidden_dim, self.hidden_dim), # W2
                       (1, self.out_dim)] # b2
        
        shapes_NN = [(self.in_dim, self.hidden_dim), # W1
                       (1, self.hidden_dim), # b1
                       (self.hidden_dim, self.out_dim), # W2
                       (1, self.out_dim)] # b2
        
        self.RESNET = RESNET
        self.name = name
        self.sampling_region = sampling_region 
        self.sampling_points = sampling_points
        

        
        if(self.RESNET):
            self.shapes = shapes_RESNET
        else:
            self.shapes = shapes_NN
            
        def get_V_indexes(shapes):
            self.V_indexes = []
            s = 0
            for (z0, z1) in shapes:
                self.V_indexes.append((s, s+(z0*z1)))
                s += z0*z1
        get_V_indexes(self.shapes)

    def break_super_vector(self):
        # Random initialization
        if(self.RESNET):
            self.V, (self.W, self.b, self.W1, self.b1, self.W2, self.b2) = CatVariable(self.shapes)
            self.weights = [self.W, self.W1, self.W2] 
        else:   
            self.V, (self.W1, self.b1, self.W2, self.b2) = CatVariable(self.shapes)
            self.weights = [self.W1, self.W2]  
        # s_tweaked needed for optimization step.
    
    def set_threshold_holder(self):
        self.threshold = tf.Variable(tf.zeros(shape = (), dtype = tf.float64), name = "threshold_holder")
        
    def inference(self):

        # define inference
        
        nonlinear_part = (act(tf.linalg.matmul(self.inputs, self.W1) + self.b1))
        nonlinear_part = (tf.linalg.matmul(nonlinear_part, self.W2) + self.b2) ## zeros
        
        if(self.RESNET):
            linear_part = tf.linalg.matmul(self.inputs, self.W + self.b)
            self.pred = linear_part + nonlinear_part
        else:
            self.pred = nonlinear_part
            
    def two_phase_optimization(self, split = 0.90):
        self.two_phase_training = True
        self.second_phase_start = int(split * self.training_epochs)
    
    
    def operations(self):
        with tf.device('/GPU:0'):
            super(experiment_phase2, self).operations()

            ## training 
            self.set_threshold_holder()

            '''
            # newton's direction
            self.newton_ops = NewtonsMethod_avoid_saddle_points_modified_direction(self.cost, 
                                                                           self.V, 
                                                                           0)
            '''
            # sparse T_k

            self.a0 = tf.constant(0, dtype = tf.float64)
            self.b0 = tf.constant(10, dtype = tf.float64)
            self.ls_threshold = tf.constant(1e-8, dtype = tf.float64)
        
            # q_holder for lanczos operation
            self.num_of_params = int(self.V.shape[0])
            self.q_holder = tf.Variable(tf.zeros(shape = [self.num_of_params, K+2], dtype = dtypes.float64))

    def saver(self):
        self.to_save = {'cost_hist':self.cost_hist, 
                        'loss_hist':self.loss_hist, 
                        'grad_hist':self.dc_dw_autodiff_norm_hist,
                        'step_length_hist':self.step_length_hist
                        }
        
        with open(self.name+'.pkl','wb') as f:
            pickle.dump(self.to_save, f)


        
    # sampling the cost function; call in line_saerch
    def compute_cost(self, V):
        
        v_indexes = self.V_indexes
        shapes = self.shapes
        
        W1 = tf.reshape(V[v_indexes[0][0]: v_indexes[0][1]], shapes[0])
        b1 = tf.reshape(V[v_indexes[1][0]: v_indexes[1][1]], shapes[1])          
        W2 = tf.reshape(V[v_indexes[2][0]: v_indexes[2][1]], shapes[2])
        b2 = tf.reshape(V[v_indexes[3][0]: v_indexes[3][1]], shapes[3])

        pred = (act(math_ops.matmul(self.inputs, W1) + b1))
        pred = (math_ops.matmul(pred, W2) + b2)

        return math_ops.reduce_mean(1/2. * math_ops.square(pred-self.Y)) + \
                self.regularization_coeff * math_ops.reduce_mean(math_ops.square(V))
    
    
    def train(self, k):

        with tf.Session() as sess:
            
            sess.run(self.init)
            
            run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            feed_dict_ = {self.X: self.train_X, self.Y: self.train_Y}
            
            self.lanczos_ops = get_lanczos_ops(K, self.cost, self.V, self.q_holder)
            self.gradient = sess.run(self.dc_dw_autodiff, feed_dict = feed_dict_)[0]
            
            with tf.device('/GPU:0'):

                for epoch in range(self.training_epochs):
                    Q = self.lanczos_ops[0]
                    alpha = self.lanczos_ops[1]
                    beta = self.lanczos_ops[2]
                    
                    self.Q_val = sess.run(Q, feed_dict = feed_dict_, options=run_options, run_metadata=run_metadata)
                    self.alpha_val = sess.run(alpha, feed_dict = feed_dict_, options=run_options, run_metadata=run_metadata)
                    
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('train_krylov_timeline.json', 'w') as f:
                        f.write(ctf)

                    self.beta_val = sess.run(beta, feed_dict = feed_dict_)                    
                    
                    T_k = tridiagonal(alpha, beta, beta)

                    ## solve the eigen problem for T_k: 
                    ## Tridiagonal Symmetric matrix. Could be solved in O(kN) time.
                    ## using Tensorflow's default solver for now, ( which is O(N^3) )
                    W, V = tf.linalg.eigh(T_k)
                    QV = math_ops.matmul(Q, V)
                    W_abs_inverse =  tf.diag(1./math_ops.abs(W))

                    self.QV = sess.run(QV, feed_dict = feed_dict_)
                    self.W_abs_inverse = sess.run(W_abs_inverse, feed_dict = feed_dict_)

                    # compute direction
                    dx = tf.reshape(math_ops.matmul(math_ops.matmul(math_ops.matmul(QV, 
                                                                                    W_abs_inverse), 
                                                                    QV, transpose_b = True),
                                                    -array_ops.expand_dims(self.dc_dw_autodiff[0], -1)), [-1])

                    # spit out line_search results.

                    a1 = (2-GOLDENRATIO) * self.b0 + (GOLDENRATIO-1) * self.a0
                    b1 = (2-GOLDENRATIO) * self.a0 + (GOLDENRATIO-1) * self.b0
                    fa1 = self.compute_cost(self.V + dx * a1)
                    fb1 = self.compute_cost(self.V + dx * b1)

                    i, a0, a1, b1, b0, fa1, fb1, changed, ls_threshold = line_search(self.compute_cost, 
                                                                                                 self.V, 
                                                                                                 dx, 
                                                                                                 10, 
                                                                                                 tf.constant(1e-8, dtype = dtypes.float64),
                                                                                                 self.a0,
                                                                                                 self.b0,
                                                                                                 a1,
                                                                                                 b1,
                                                                                                 fa1,
                                                                                                 fb1)

                    # finally, update V
                    step_length = (a0 + b0)/2.


                    update_V = tf.assign(self.V, self.V + dx * step_length)
                    sess.run(update_V, feed_dict = feed_dict_)

                    if (epoch+1) % LOGSTEP_PHASE2 == 0:
                        c = sess.run(self.cost, feed_dict = feed_dict_)
                        self.cost_hist.append(c)

                        l = sess.run(self.loss, feed_dict = feed_dict_)
                        self.loss_hist.append(l)


                        self.gradient_autodiff_norm = sess.run(self.dc_dw_autodiff_norm, feed_dict = feed_dict_)

                        # store values
                        self.dc_dw_autodiff_norm_hist.append(self.gradient_autodiff_norm)

                        # store step length
                        dx_norm = sess.run(util.l2norm(dx * step_length)/self.num_of_params, feed_dict = feed_dict_)
                        self.step_length_hist.append(dx_norm)
                        self.saver()







def main():
  ## Black box optimizer
  exp_BB = experiment_phase1(dim = DIM_IN, 
                   hidden_dim = DIM_HIDDEN, 
                   Ndata = NDATA,
                   dim_y = DIM_Y, 
                   train_epochs = TRAIN_EPOCHS_PHASE1,
                   display_step = DISPLAY_STEP_PHASE1,
                   regularization_coeff = REGULARIZATION_PARAM,
                   name = MODEL_SAVE_NAME_PHASE1,
                   RESNET = RESNET_NN)
  exp_BB.build()

  exp_BB.set_optimizer( [tf.train.AdamOptimizer(2e-4).minimize(exp_BB.cost)], 'phase1_optimizer')
  #exp_BB.set_optimizer( [tf.train.GradientDescentOptimizer(1e-2).minimize(exp_BB.cost)], 'phase1_optimizer')
  exp_BB.init()
  exp_BB.train(k = K)


  ## Phase II
  exp_KRYLOV = experiment_phase2(dim = DIM_IN, 
                               hidden_dim = DIM_HIDDEN, 
                               Ndata = NDATA,
                               dim_y = DIM_Y, 
                               train_epochs = TRAIN_EPOCHS_PHASE2, 
                               display_step = DISPLAY_STEP_PHASE2,
                               regularization_coeff = REGULARIZATION_PARAM,
                               RESNET = RESNET_NN,
                               sampling_region = SAMPLING_REGION,
                               sampling_points = SAMPLING_POINTS,
                               name = MODEL_SAVE_NAME_PHASE2,
                               generate_data = high_d_data_NN)


  if(TWO_PHASE):
      exp2 = exp_KRYLOV

      exp2.V, (exp2.W1, exp2.b1, exp2.W2, exp2.b2) = utils.CatVariable_Specified(exp_BB.shapes, exp_BB.V_val)
      exp2.shapes = exp_BB.shapes

      exp2.weights = [exp2.W1, exp2.W2]

  else:
      exp_KRYLOV.break_super_vector()

  exp_KRYLOV.inference()
  exp_KRYLOV.operations()

  if(TWO_PHASE):
      exp2.cost_hist = list(exp_BB.cost_hist)
      exp2.loss_hist = list(exp_BB.loss_hist)
      exp2.dc_dw_autodiff_norm_hist = list(exp_BB.dc_dw_autodiff_norm_hist)

  exp_KRYLOV.two_phase_optimization(0.00)

  exp_KRYLOV.init()
  # k <= K
  exp_KRYLOV.train(k = int( K))

if __name__ == "__main__":
    main()

      
    


