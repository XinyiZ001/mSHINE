import numpy as np
import tensorflow as tf

margin = 5.0
epsilon = 1e-10
def build_edge_type_metric(type_indicator, type_select_tensor, type_based_metric, metric_len, name):
    """
    :param type_indicator: (?, )
    :param type_select_tensor: constant (1, num_type)
    :param type_based_metric: variable (1, metric_shape)
    :param batch_size:
    :param name:
    :return: (?, metric_shape)
    """
    with tf.compat.v1.variable_scope(name):
        batch_size = tf.shape(type_indicator)[0]
        type_indicator = tf.reshape(type_indicator, shape=[-1,1])  # (?, 1)
        type_split_mask = tf.equal(type_indicator, type_select_tensor)

        type_based_metric_tile = tf.tile(type_based_metric, multiples=[batch_size]+[1]*(metric_len-1))
        type_based_metric_all = tf.boolean_mask(type_based_metric_tile, type_split_mask)
    return type_based_metric_all

def build_k_edge_type_metric(type_indicator, type_select_tensor, type_based_metric, metric_len, name):
    """
    :param type_indicator: (?, )
    :param type_select_tensor: constant (1, num_type)
    :param type_based_metric: variable (1, metric_shape)
    :param batch_size:
    :param name:
    :return: (?, metric_shape)
    """
    with tf.compat.v1.variable_scope(name):
        batch_size = tf.shape(type_indicator)[0]
        type_indicator = tf.reshape(type_indicator, shape=[-1,1])  # (?, 1)
        type_split_mask = tf.equal(type_indicator, type_select_tensor)

        type_based_metric_tile = tf.tile(type_based_metric, multiples=[batch_size]+[1]*(metric_len-1))
        type_based_metric_all = tf.boolean_mask(type_based_metric_tile, type_split_mask)
    return type_based_metric_all


def build_type_metric(metapath_type_indicator, node_type_indicator, metapath_type_select_tensor, node_type_select_tensor, type_based_metric, metric_len, name):
    """
    :param metapath_type_indicator: (?, )
    :param node_type_indicator: (?, )
    :param metapath_type_select_tensor: (1, num_metapath_type)
    :param node_type_select_tensor: (1, num_node_type)
    :param type_based_metric: (1, num_node_type, num_metapath_type, 1, emb_size)
    :param metric_len:
    :param name:
    :return:
    """
    with tf.compat.v1.variable_scope(name):
        batch_size = tf.shape(metapath_type_indicator)[0]
        metapath_type_indicator = tf.reshape(metapath_type_indicator, shape=[-1,1])  # (?, 1)
        node_type_indicator = tf.reshape(node_type_indicator, shape=[-1,1]) # (?, 1)
        metapath_type_split_mask = tf.equal(metapath_type_indicator, metapath_type_select_tensor)  # (?, num_metapath)
        node_type_split_mask = tf.equal(node_type_indicator, node_type_select_tensor)  # (?, num_node)

        type_based_metric_tile = tf.tile(type_based_metric, multiples=[batch_size]+[1]*(metric_len-1))
        type_based_metric_all = tf.boolean_mask(type_based_metric_tile, node_type_split_mask)
        type_based_metric_all = tf.boolean_mask(type_based_metric_all, metapath_type_split_mask)
    return type_based_metric_all


def edge_representation(u_emb,v_emb,mode =1):
    """
    :param u_emb:
    :param v_emb:
    :return:
    """
    # mode 1: hadamard-product
    # mode 2: outer-product
    # mode 3: deduction
    # mode 4: addition
    if mode == 1:
        return u_emb * v_emb
    elif mode == 2:
        return u_emb + v_emb
    elif mode == 3:
        return (u_emb - v_emb) ** 2
    elif mode == 4:
        return (u_emb + v_emb) ** 2
    else:
        return tf.concat([(u_emb-v_emb)**2,(u_emb+v_emb)**2],axis=2)

def normalization(x, axis):
    norm = tf.norm(x, axis=axis, keep_dims=True)+epsilon
    return x/norm


def loss_undirected_path(input_embedding,
                         state_embedding,
                         output_embedding,
                         h_node_index,
                         m_node_index,
                         t_node_index,
                         t_node_noise,
                         t_node_noise_mask,
                         metapath_type_metric_h,
                         metapath_type_metric_x,
                         metapath_type_metric_y,
                         edge_type_metric,
                         to_node_m_emb_bias,
                         node_emb_size,
                         Wxh, Whh, Wrh,
                         name='loss_function'):

    """
    :param input_embedding: (N, d)
    :param state_embedding:（N, d）
    :param output_embedding:（N, d）
    :param h_node_index: (?,)
    :param m_node_index: (?,)
    :param t_node_index: (?,)
    :param t_node_noise: (?, num_neg_sample)
    :param num_node_type: int
    :param t_node_type: (?,)
    :param t_node_noise_mask: (?, num_neg_sample)
    :param metapath_type_metric: (?, 1, d)
    :param edge_type_metric: (?, 1, d)
    :param to_node_t_emb_bias: (1, 1, d)
    :param to_node_m_emb_bias: (1, 1, d)
    :param edge_type_bias: (1, 1, d)
    :param node_emb_size: int
    :param Wxh: (d, d)
    :param Whh: (d, d)
    :param Whx: (d, d)
    :param Wrh: (d, d)
    :param Wrx: (d, d)
    :param name:
    :return:
    """
    with tf.compat.v1.variable_scope(name):
        with tf.compat.v1.variable_scope('embedding_prepare'):
            h_state_embedding = tf.nn.embedding_lookup(state_embedding, h_node_index)[:, tf.newaxis, :]
            m_input_embedding = tf.nn.embedding_lookup(input_embedding, m_node_index)[:, tf.newaxis, :]
            m_state_embedding = tf.nn.embedding_lookup(state_embedding, m_node_index)[:, tf.newaxis, :]
            t_output_embedding = tf.nn.embedding_lookup(output_embedding, t_node_index)[:, tf.newaxis, :]  # (?, 1, node_emb)
            t_noise_embedding = tf.nn.embedding_lookup(output_embedding, t_node_noise)  # (?, num_neg, node_emb)

        with tf.compat.v1.variable_scope('regular_loss_ori'):
            loss_regular_pos_ori = regular_loss(h_state_embedding) \
                                   + regular_loss(m_input_embedding) \
                                   + regular_loss(t_output_embedding) \
                                   + regular_loss(m_state_embedding)  # (?, 1)
            loss_regular_neg_ori = regular_loss(t_noise_embedding)  # (?, num_neg)

        with tf.compat.v1.variable_scope('metapath_filter'):
            h_state_embedding = h_state_embedding * metapath_type_metric_h  # (?, 1, node_emb)
            m_input_embedding = m_input_embedding * metapath_type_metric_x
            m_state_embedding = m_state_embedding * metapath_type_metric_h
            t_output_embedding = t_output_embedding * metapath_type_metric_y
            t_noise_embedding = t_noise_embedding * metapath_type_metric_y

        with tf.compat.v1.variable_scope('regular_loss_metapath'):
            loss_regular_pos_metapath = regular_loss(t_output_embedding) \
                                        + regular_loss(h_state_embedding) \
                                        + regular_loss(m_input_embedding) \
                                        + regular_loss(m_state_embedding)  # (?, 1)
            loss_regular_neg_metapath = regular_loss(t_noise_embedding)  # (?, num_neg)

        with tf.compat.v1.variable_scope('to_state_node_emb'):
            h_input_embedding = tf.reshape(
                tf.matmul(tf.reshape(h_state_embedding, [-1, node_emb_size]), Whh, name='h_to_node_h_emb'),
                [-1, 1, node_emb_size])
            m_input_embedding = tf.reshape(
                tf.matmul(tf.reshape(m_input_embedding, [-1, node_emb_size]), Wxh, name='x_to_node_h_emb'),
                [-1, 1, node_emb_size])
            r_input_embedding = tf.reshape(
                tf.matmul(tf.reshape(edge_type_metric, [-1, node_emb_size]), Wrh, name='r_to_node_h_emb'),
                [-1, 1, node_emb_size])
            m_state_embedding_predict = tf.tanh(
                h_input_embedding + m_input_embedding + r_input_embedding + to_node_m_emb_bias)

        with tf.compat.v1.variable_scope('loss_state_embedding'):
            m_state_embedding_label = tf.stop_gradient(m_state_embedding_predict)
            m_state_embedding_loss = tf.reduce_mean(tf.pow(m_state_embedding_label - m_state_embedding, 2), axis=-1)

        with tf.compat.v1.variable_scope('loss_calculation'):
            true_logits = tf.reduce_sum(m_state_embedding_predict * t_output_embedding, axis= -1)
            sampled_logits = tf.reduce_sum(m_state_embedding_predict * t_noise_embedding,axis=-1)
            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(true_logits), logits=true_logits)

            sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(sampled_logits), logits=sampled_logits)* t_node_noise_mask


        loss_pos_backward = tf.reduce_mean(tf.cast(true_xent, dtype=tf.float64))
        loss_neg_backward = tf.reduce_mean(tf.cast(tf.reduce_mean(sampled_xent,axis=-1,keepdims=True), dtype=tf.float64))
        m_state_embedding_loss_backward = tf.reduce_mean(tf.cast(m_state_embedding_loss, dtype=tf.float64))

        loss_print = true_xent + \
                     tf.reduce_mean(sampled_xent, axis=-1, keepdims=True) + \
                     m_state_embedding_loss

        loss = loss_pos_backward + loss_neg_backward + m_state_embedding_loss_backward

    return loss, loss_pos_backward, loss_neg_backward, m_state_embedding_loss_backward, loss_regular_pos_ori + loss_regular_neg_ori + regular_loss(m_state_embedding_predict), loss_regular_pos_metapath+loss_regular_neg_metapath,loss_print

def regular_loss(t_node_emb,limit=1):
    norm = tf.reduce_sum(t_node_emb**2, axis=-1)  # (?,)
    loss_regular = tf.reduce_mean(tf.reduce_max(tf.maximum(norm-limit, 0),axis=-1))
    return tf.cast(loss_regular,dtype=tf.float64)