import argparse,os
import numpy as np
import tensorflow as tf
import pickle
import time

from src.tf_model import mSHINE
import src.tf_data_utils as tf_data_utils

global config

def output_dir_prepare(args):
    experi_data_dir = os.path.join('data',args.graph_name, 'experi_data')
    if not os.path.exists(experi_data_dir):
        os.mkdir(experi_data_dir)
    output_dir = experi_data_dir
    record_dir = os.path.join(output_dir, 'record')
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    node_emb_dir = os.path.join(output_dir, 'node_emb')
    if not os.path.exists(node_emb_dir):
        os.mkdir(node_emb_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return experi_data_dir, output_dir, record_dir, node_emb_dir, log_dir


def parse_args():
    '''
    Parses the mSHINE arguments.
    '''
    parser = argparse.ArgumentParser(description="Run mSHINE.")

    parser.add_argument('--dimensions', type=int, default= 128, help='Number of dimensions. Default is 128.')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size. Default is 30.')

    parser.add_argument('--record_iter', type=int, default=50, help='Record iter. Default is 50')

    parser.add_argument('--neg_sample', type=int, default=5, help='Num of neg samples. Default is 5.')

    parser.add_argument('--graph_name', type=str, default='cora', help='Name of dataset')

    parser.add_argument('--restore', type=bool, default=False, help='If restore trained model')

    parser.add_argument('--restore_iter', type=int, default=20, help='Restore trained model which is trained for n iters')

    parser.add_argument('--iter', default=1000, type=int, help='Number of epochs to train')

    return parser.parse_args()


def save_metric(trans_metric_, iter, record_dir):
    trans_metric = {
        'metapath_type_metric_x': trans_metric_[0],
        'metapath_type_metric_h': trans_metric_[1],
        'metapath_type_metric_y': trans_metric_[2],
        'edge_type_metric': trans_metric_[3],
        'edge_type_bias': trans_metric_[4],
        'Wxh': trans_metric_[5],
        'Whh': trans_metric_[6],
        'Wrh': trans_metric_[7],
        'to_node_m_emb_bias': trans_metric_[8]
    }

    with open(os.path.join(record_dir, '_'.join(
            ['trans_metric', str(iter), str(args.dimensions)])), 'wb') as f:
        pickle.dump(trans_metric, f)


def save_emb(sess, model, epoch,  node_emb_dir):
    i_embedding = sess.run(model.input_embedding)
    s_embedding = sess.run(model.state_embedding)
    o_embedding = sess.run(model.output_embedding)
    with open(os.path.join(node_emb_dir, '_'.join(
            [args.graph_name, str(epoch), str(args.dimensions) + '.emb'])),
              'wb') as f:
        pickle.dump({
            'i_embedding': i_embedding,
            's_embedding': s_embedding,
            'o_embedding': o_embedding
        }, f)


def build_feed_dict(model, line):
    input_list = list(zip(*line))
    feed_dict = dict()

    feed_dict[model.h_node_type], feed_dict[model.h_node_index], feed_dict[model.m_node_type], \
    feed_dict[model.m_node_index], feed_dict[model.t_node_type], feed_dict[model.t_node_index], \
    feed_dict[model.edge_type], feed_dict[model.metapath_type] = input_list
    return feed_dict


def main(args):
    ##  Prepare output_dir
    experi_data_dir, output_dir, record_dir, node_emb_dir, log_dir= output_dir_prepare(args)

    ##  Data preparation
    input_dir = os.path.join('data',args.graph_name)
    Data_Sampler = tf_data_utils.Sampler()
    Data_Sampler.edge_list_preparison(os.path.join(input_dir, args.graph_name+'.hin'))

    data_info = pickle.load(open(os.path.join(input_dir, args.graph_name + '_info_dict.p'), 'rb'))
    node_mappings = pickle.load(open(os.path.join(input_dir, args.graph_name + '_id_to_index.p'), 'rb'))
    Hin_dict = pickle.load(open(os.path.join(input_dir, args.graph_name + '.HIN_dict'), 'rb'))
    config = tf_data_utils.read_config(os.path.join(input_dir, args.graph_name +'.config'))

    Data_Sampler.init(config,Hin_dict, node_mappings, data_info)
    print('Network Spec:')
    print('NODE TYPES: {}'.format(config['nodes']))
    print('NUM of Meta-paths: {}'.format(len(config['metapath_types'])))


    ##  Build Model
    model = mSHINE(
        num_node = data_info['num_node'],
        node_emb_size=args.dimensions,
        neg_num_sample = args.neg_sample,
        num_edge_type = len(config['edge_types']),
        num_metapath_type = len(config['metapath_types']),
        num_node_type = len(config['nodes']),
        off_set_min = data_info['off_set_min'],
        num_node_each_type = data_info['num_node_each_type'],
        learning_rate = 0.5,
    )

    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    ##  Start Training
    with tf.compat.v1.Session(graph=model.graph) as sess:
        DataTester = tf_data_utils.DataTest(Data_Sampler=Data_Sampler, batch_size=args.batch_size)
        iterator, next_element = DataTester.data_read_in()
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        iter_start = 0
        loss_epoch = []
        if args.restore:
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, os.path.join(log_dir, args.graph_name + '-' + str(args.restore_iter)))
            iter_start = args.restore_iter + 1

        for epoch in range(iter_start, args.iter):
            loss_list = [];iter_sub = -1;start_time = time.time()
            sess.run(iterator.initializer)
            while True:
                try:
                    iter_sub += 1
                    line = sess.run(next_element)
                    DataTester.num_processed_edge += args.batch_size
                    feed_dict = build_feed_dict(model, line)

                    _, loss_tuple, trans_metric_, global_step, learning_rate = sess.run(
                        [
                            model.optimizer,
                            model.loss_print,
                            model.trans_metric,
                            model.global_step,
                            model.current_learning_step],
                        feed_dict=feed_dict)
                    loss, pos_loss, neg_loss, regular_loss_all, regular_loss_all_metapath, loss_print, m_state_embedding_loss_backward = loss_tuple
                    loss_list.append(loss)

                    if iter_sub % 1000 == 0:
                        print('Iter epoch:{}, sub:{}, time_dur: {}'.format(epoch, iter_sub, time.time()-start_time))
                        start_time = time.time()
                        print('     loss: {:.4f}, step: {}, learning_rate: {}\n'
                              '     pos_loss: {:.4f}, neg_loss: {:.4f}, state:{:.4f}, regular_loss: {:.4f}, regular_loss_metapath:{:.4f}'.format(
                                loss, global_step, learning_rate,
                            pos_loss, neg_loss, m_state_embedding_loss_backward,regular_loss_all, regular_loss_all_metapath))

                    if iter_sub % 1000 == 0 and epoch % args.record_iter == 0:
                        save_metric(trans_metric_, epoch, record_dir)

                except:
                    loss_epoch.append(np.mean(np.array(loss_list)))
                    print('########## Loss For Epoch {}: {}'.format(epoch, np.mean(np.array(loss_list))))
                    if epoch % args.record_iter == 0:
                        model.saver.save(sess, os.path.join(log_dir, args.graph_name), global_step=epoch)

                    if epoch % args.record_iter == 0:
                        save_metric(trans_metric_, epoch, record_dir)
                        save_emb(sess, model, epoch,  node_emb_dir)
                    break

if __name__ == "__main__":
    args = parse_args()
    main(args)
