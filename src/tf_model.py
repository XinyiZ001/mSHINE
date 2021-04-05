import tensorflow as tf
import numpy as np
from src.tf_model_utils import *

epsilon = 1e-10

class mSHINE(object):
    '''
    mSHINE structure
    '''
    def __init__(self,num_node, node_emb_size, neg_num_sample, num_edge_type,num_metapath_type, num_node_type, off_set_min,num_node_each_type,learning_rate):
        self.num_node = num_node
        self.neg_num_sample = neg_num_sample
        self.node_emb_size = node_emb_size
        self.num_edge_type = num_edge_type
        self.num_metapath_type = num_metapath_type
        self.num_node_type = num_node_type
        self.off_set_min = off_set_min
        self.num_node_each_type = num_node_each_type
        self.learning_rate = learning_rate
        self.regular_loss_p = 0.005
        self.build_structure()

    def prepare_metric(self):
        with tf.compat.v1.variable_scope('prepare_edge_loss_ratio'):
            self.edge_type_select_tensor = tf.constant(
                np.reshape(np.array(list(range(self.num_edge_type))), newshape=[1, -1]), dtype=tf.int32)  # (1, num_type)

        with tf.compat.v1.variable_scope('prepare_metapath_loss_ratio'):
            self.metapath_type_select_tensor = tf.constant(
                np.reshape(np.array(list(range(self.num_metapath_type))), newshape=[1, -1]), dtype=tf.int32)  # (1, num_type)

        with tf.compat.v1.variable_scope('prepare_filters'):
            metapath_type_metric_x = tf.Variable(
                tf.ones(shape=[self.num_metapath_type, self.node_emb_size]),
                name='metapath_type_metric_x')  # (num_node_type, num_metapath_type, emb_size)
            self.metapath_type_metric_x = metapath_type_metric_x[tf.newaxis, :, tf.newaxis, :]  # (1, num_metapath_type, 1, emb_size)

            metapath_type_metric_h = tf.Variable(
                tf.ones(shape=[self.num_metapath_type, self.node_emb_size]),
                name='metapath_type_metric_h')  # (num_node_type, num_metapath_type, emb_size)
            self.metapath_type_metric_h = metapath_type_metric_h[tf.newaxis, :, tf.newaxis,
                                          :]  # (1, num_metapath_type, 1, emb_size)

            metapath_type_metric_y = tf.Variable(
                tf.ones(shape=[self.num_metapath_type, self.node_emb_size]),
                name='metapath_type_metric_y')  # (num_node_type, num_metapath_type, emb_size)
            self.metapath_type_metric_y = metapath_type_metric_y[tf.newaxis, :, tf.newaxis,
                                          :]  # (1, num_metapath_type, 1, emb_size)

            edge_type_metric = tf.Variable(
                tf.random.normal(shape=[self.num_edge_type, 1, self.node_emb_size]),
                name='edge_type_metric')  # (num_node_type, 1, emb_size)
            self.edge_type_metric = edge_type_metric[tf.newaxis, :, :, :]  # (1, num_edge_type, 1, emb_size)

            edge_type_bias = tf.Variable(
                tf.zeros(shape=[1, self.node_emb_size]),
                name='edge_type_bias')  # (1, emb_size)
            self.edge_type_bias = edge_type_bias[tf.newaxis, :, :]

            self.to_node_m_emb_bias = tf.Variable(tf.random.normal(shape=[1, 1, self.node_emb_size]),
                                                  name='to_node_m_emb_bias')
            self.to_node_t_emb_bias = tf.Variable(tf.random.normal(shape=[1, 1, self.node_emb_size]),
                                                  name='to_node_t_emb_bias')

        with tf.compat.v1.variable_scope('prepare_node'):
            self.node_type_select_tensor = tf.constant(
                np.reshape(np.array(list(range(self.num_node_type))), newshape=[1, -1]), dtype = tf.int32)  # (1, num_type)
            self.num_node_each_type = tf.constant(
                np.reshape(self.num_node_each_type, newshape=[1, -1, 1]), dtype=tf.float32)  # (1, num_node_type, 1)
            self.num_node_min = tf.constant(
                np.reshape(self.off_set_min, newshape=[1, -1, 1]), tf.float32)  # (1, num_node_type, 1)

    def build_structure(self):

        graph = tf.compat.v1.Graph()
        with graph.as_default():

            self.prepare_metric()

            h_node_index = tf.compat.v1.placeholder(tf.int32, shape=([None, ]), name= 'input_h_node_index')
            m_node_index = tf.compat.v1.placeholder(tf.int32, shape=([None,]), name= 'input_m_node_index')
            t_node_index = tf.compat.v1.placeholder(tf.int32, shape=([None, ]), name='input_t_node_index')
            h_node_type = tf.compat.v1.placeholder(tf.int32, shape=([None, ]), name='input_h_node_type')
            m_node_type = tf.compat.v1.placeholder(tf.int32, shape=([None, ]), name='input_m_node_type')
            t_node_type = tf.compat.v1.placeholder(tf.int32, shape=([None, ]), name='input_t_node_type')

            edge_type = tf.compat.v1.placeholder(tf.int32, shape=([None, ]), name='input_edge_type')
            metapath_type = tf.compat.v1.placeholder(tf.int32, shape=([None,]), name='input_metapath_type')

            self.h_node_index = h_node_index
            self.h_node_type = h_node_type
            self.m_node_index = m_node_index
            self.m_node_type = m_node_type
            self.t_node_index = t_node_index
            self.t_node_type = t_node_type

            self.edge_type = edge_type
            self.metapath_type = metapath_type

            input_embedding = tf.Variable(
                tf.random.normal(shape=[self.num_node + 1, self.node_emb_size], mean=0.0, stddev=0.1),
                name='input_embedding')
            state_embedding = tf.Variable(tf.zeros(shape=[self.num_node + 1, self.node_emb_size]),
                                          name='state_embedding')
            output_embedding = tf.Variable(tf.zeros(shape=[self.num_node + 1, self.node_emb_size]),
                                          name='output_embedding')

            Wxh = tf.Variable(
                tf.random.normal(shape=[self.node_emb_size, self.node_emb_size], mean=0.0, stddev=0.01),
                name='W_xh')
            Whh = tf.Variable(
                tf.random.normal(shape=[self.node_emb_size, self.node_emb_size], mean=0.0, stddev=0.01),
                name='W_hh')
            Wrh = tf.Variable(
                tf.random.normal(shape=[self.node_emb_size, self.node_emb_size], mean=0.0, stddev=0.01),
                name='W_hx')

            self.input_embedding = input_embedding
            self.state_embedding = state_embedding
            self.output_embedding = output_embedding
            self.trans_metric = (self.metapath_type_metric_x,
                                 self.metapath_type_metric_h,
                                 self.metapath_type_metric_y,
                                 self.edge_type_metric,
                                 self.edge_type_bias,
                                 Wxh,
                                 Whh,
                                 Wrh,
                                 self.to_node_m_emb_bias)

            edge_type_metric_all = build_edge_type_metric(
                type_indicator=edge_type,
                type_select_tensor=self.edge_type_select_tensor,
                type_based_metric=self.edge_type_metric,
                metric_len=4,
                name='build_edge_metric')  # (?, 1, node_emb_size)

            metapath_type_metric_all_x = build_edge_type_metric(
                type_indicator=metapath_type,
                type_select_tensor=self.metapath_type_select_tensor,
                type_based_metric=self.metapath_type_metric_x,
                metric_len=4,
                name='build_metapath_metric_x')  # (?, 1, node_emb_size)

            metapath_type_metric_all_h = build_edge_type_metric(
                type_indicator=metapath_type,
                type_select_tensor=self.metapath_type_select_tensor,
                type_based_metric=self.metapath_type_metric_h,
                metric_len=4,
                name='build_metapath_metric_h')  # (?, 1, node_emb_size)

            metapath_type_metric_all_y = build_edge_type_metric(
                type_indicator=metapath_type,
                type_select_tensor=self.metapath_type_select_tensor,
                type_based_metric=self.metapath_type_metric_y,
                metric_len=4,
                name='build_metapath_metric_y')  # (?, 1, node_emb_size)

            t_node_noise, t_node_noise_mask = self.noise_sampler(
                t_node_index = t_node_index,
                t_node_type = t_node_type,
                neg_num_sampled = self.neg_num_sample,
                num_node = self.num_node,
                name='noise_sampler')

            loss_distance, pos_loss, neg_loss, m_state_embedding_loss_backward, regular_loss_all, regular_loss_all_metapath,loss_print = loss_undirected_path(
                input_embedding = input_embedding,
                state_embedding = state_embedding,
                output_embedding = output_embedding,
                h_node_index = h_node_index,
                m_node_index = m_node_index,
                t_node_index = t_node_index,
                t_node_noise = t_node_noise,
                t_node_noise_mask = t_node_noise_mask,
                Wxh = Wxh,
                Whh = Whh,
                Wrh = Wrh,
                to_node_m_emb_bias=self.to_node_m_emb_bias,
                metapath_type_metric_x = metapath_type_metric_all_x,
                metapath_type_metric_h = metapath_type_metric_all_h,
                metapath_type_metric_y = metapath_type_metric_all_y,
                edge_type_metric = edge_type_metric_all,
                node_emb_size= self.node_emb_size,
                name='loss_function',)

            loss = loss_distance + regular_loss_all*self.regular_loss_p + regular_loss_all_metapath*self.regular_loss_p
            node_emb_var_list = []
            metapath_var_list = []
            edge_var_list = []
            filters=[]
            for var in tf.compat.v1.global_variables():
                if var.op.name.startswith('prepare_filters'):
                    filters.append(var)
                elif var.op.name.startswith('metapath'):
                    metapath_var_list.append(var)
                elif var.op.name.startswith('edge'):
                    edge_var_list.append(var)
                elif var.op.name.startswith('node'):
                    node_emb_var_list.append(var)
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate,
                                                       global_step, 5*10e6, 0.95,
                                                       staircase=True)  # linear decay over time

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
            optimizer_filters = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)

            self.optimizer_filters = optimizer_filters.minimize(loss, global_step=global_step, var_list=filters)
            self.optimizer = optimizer.minimize(loss,global_step=global_step)

            self.current_learning_step = learning_rate
            self.global_step = global_step
            self.loss_print = (loss,pos_loss,neg_loss,regular_loss_all,regular_loss_all_metapath,loss_print,m_state_embedding_loss_backward)

            saver = tf.compat.v1.train.Saver(max_to_keep=50)
            self.saver = saver
            self.graph = graph


    def noise_sampler(self, t_node_index, t_node_type, neg_num_sampled, num_node, name):
        """
        :param u_node_index: (?, )
        :param v_node_index: (?, )
        :param u_node_type: (?, )
        :param v_node_type: (?, )
        :param batch_size: int
        :param neg_num_sampled: int
        :param num_node: constant
        :return: (?, C)
        """

        with tf.compat.v1.variable_scope(name):
            # neighbors = tf.transpose(neighbors,perm=[1,0])
            batch_size = tf.shape(t_node_type)[0]
            neg_samples_ori = tf.random.uniform(
                shape=[batch_size, neg_num_sampled],
                minval=0,
                maxval=num_node,
                dtype=tf.dtypes.int32,
                seed=None,
                name='noise_sample_ori')  # (?, C)

            rescale_width_t = build_edge_type_metric(
                type_indicator=t_node_type,
                type_select_tensor=self.node_type_select_tensor,
                type_based_metric=self.num_node_each_type,
                metric_len=3,
                name='build_rescale_width_t')  # (?, num_node_type)

            rescale_min_t = build_edge_type_metric(
                type_indicator=t_node_type,
                type_select_tensor=self.node_type_select_tensor,
                type_based_metric=self.num_node_min,
                metric_len=3,
                name='build_rescale_min_t')  # (?, num_node_type)
            with tf.compat.v1.variable_scope('scale'):
                t_neg_samples_scaled = tf.cast(tf.cast(neg_samples_ori, dtype=tf.float32) * rescale_width_t / num_node + rescale_min_t, tf.int32)
            with tf.compat.v1.variable_scope('unequal_detect'):
                t_equal_mask = tf.equal(tf.reshape(t_node_index, shape=[-1, 1]), t_neg_samples_scaled)
            with tf.compat.v1.variable_scope('drop_equal'):
                t_neg_samples_loss_mask = tf.where(t_equal_mask,tf.zeros(tf.shape(neg_samples_ori)),tf.ones(tf.shape(neg_samples_ori)))

        return t_neg_samples_scaled, t_neg_samples_loss_mask