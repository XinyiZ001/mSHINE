import tensorflow as tf
import numpy as np
import random,ast

def read_config(conf_name):
    config = {}
    with open(conf_name) as IN:
        config['nodes'] = ast.literal_eval(IN.readline())
        config['edge_types'] = ast.literal_eval(IN.readline())
        config['metapath_types'] = ast.literal_eval(IN.readline())
        config['node_to_metapath_indicator'] = ast.literal_eval(IN.readline())
        config['metapath_dict'] = dict()
        config['metapath_to_edge_dict'] = dict()

        for meta_path_id, meta_path in enumerate(config['metapath_types']):
            for edge in meta_path:
                if not edge in config['metapath_dict']:
                    config['metapath_dict'][edge] = []
                config['metapath_dict'][edge].append(meta_path_id)
                if not meta_path_id in config['metapath_to_edge_dict']:
                    config['metapath_to_edge_dict'][meta_path_id] = []
                config['metapath_to_edge_dict'][meta_path_id].append(edge)
    return config


class Sampler(object):
    def init(self, config, Hin_dict,mappings, data_info):
        self.metapath_to_edge_type_to_edge_index_record = dict()
        self.metapath_to_edge_type_index_record = dict()
        self.metapath_to_edge_type = config['metapath_to_edge_dict']
        for metapath_idx in self.metapath_to_edge_type :
            self.metapath_to_edge_type_to_edge_index_record[metapath_idx] = dict()
            self.metapath_to_edge_type_index_record[metapath_idx] = 0
            for edge_type in self.metapath_to_edge_type [metapath_idx]:
                sub_edge = ':'.join(edge_type.split(':')[:2])
                self.metapath_to_edge_type_to_edge_index_record[metapath_idx][edge_type] = random.randint(0, self.edge_type_list_lim[sub_edge]-1)

        self.idx_to_edge_types = config['edge_types']
        self.edge_types_to_idx = {edge_type:idx for idx, edge_type in enumerate(self.idx_to_edge_types)}
        self.idx_to_node_types = config['nodes']
        self.node_types_to_idx = {node_type: idx for idx, node_type in enumerate(self.idx_to_node_types)}

        self.Hin_dict = Hin_dict
        self.mappings = mappings

        self.data_info = data_info

    def idx_record_update(self, metapath_idx, current_edge_type, edge_list_type):
        if self.metapath_to_edge_type_to_edge_index_record[metapath_idx][current_edge_type] < self.edge_type_list_lim[edge_list_type] -1:
            self.metapath_to_edge_type_to_edge_index_record[metapath_idx][current_edge_type] += 1
        else:
            self.metapath_to_edge_type_to_edge_index_record[metapath_idx][current_edge_type] = 0

        if self.metapath_to_edge_type_index_record[metapath_idx] < len(self.metapath_to_edge_type[metapath_idx])-1:
            self.metapath_to_edge_type_index_record[metapath_idx] += 1
        else:
            self.metapath_to_edge_type_index_record[metapath_idx] = 0

    def convert_to_node_idx(self, node_h_type, node_h_id, node_m_type, node_m_id, node_t_type, node_t_ids,edge_id_op, metapath_id_op):
        line = []
        node_h_type_op = self.node_types_to_idx[node_h_type]
        node_h_id_op = self.mappings['in_mapping'][node_h_type][node_h_id] + self.data_info['off_set_min'][ self.node_types_to_idx[node_h_type]]
        node_m_type_op = self.node_types_to_idx[node_m_type]
        node_m_id_op = self.mappings['in_mapping'][node_m_type][node_m_id] + self.data_info['off_set_min'][ self.node_types_to_idx[node_m_type]]
        node_t_type_op = self.node_types_to_idx[node_t_type]
        for node_t_id in node_t_ids:
            node_t_id_op = self.mappings['in_mapping'][node_t_type][node_t_id] + self.data_info['off_set_min'][ self.node_types_to_idx[node_t_type]]
            line.append((node_h_type_op, node_h_id_op, node_m_type_op, node_m_id_op, node_t_type_op, node_t_id_op,edge_id_op, metapath_id_op))
        return line

    def training_pair(self, metapath_idx, batch_size):
        current_edge_type = self.metapath_to_edge_type[metapath_idx][self.metapath_to_edge_type_index_record[metapath_idx]]
        node_h_type, node_m_type, node_t_type = current_edge_type.split(':')
        current_edge_index = self.metapath_to_edge_type_to_edge_index_record[metapath_idx][current_edge_type]
        node_h_index, node_m_index = self.edge_type_list[node_h_type+':'+node_m_type][current_edge_index]
        try:
            node_t_indexes = self.Hin_dict[node_m_type][node_m_index][node_t_type]
            node_t_indexs = [random.choice(node_t_indexes)]
            line = self.convert_to_node_idx(
                node_h_type, node_h_index, node_m_type, node_m_index, node_t_type, node_t_indexs,self.edge_types_to_idx[current_edge_type], metapath_idx)
            self.idx_record_update(metapath_idx, current_edge_type, node_h_type + ':' + node_m_type)
            return line
        except:
            self.idx_record_update(metapath_idx, current_edge_type,node_h_type+':'+node_m_type)
            return self.training_pair(metapath_idx,batch_size)

    def edge_list_preparison(self, hin_file):
        edge_type_list = dict()
        edge_type_list_lim = dict()
        lines = open(hin_file,'r').readlines()
        self.num_edge = len(lines)
        for edge in lines:
            try:
                node_a, node_b, direction, edge_type = edge.rstrip().split()
            except:
                node_a, node_b= edge.rstrip().split()
                direction = '1'
            node_a_type, node_a_id = node_a.split(':')
            node_b_type, node_b_id = node_b.split(':')
            edge_type = node_a_type+':'+node_b_type
            if edge_type not in edge_type_list:
                edge_type_list[edge_type] = []
            edge_type_list[edge_type].append((node_a_id, node_b_id))
            if direction == '1':
                edge_type = node_b_type + ':' + node_a_type
                if edge_type not in edge_type_list:
                    edge_type_list[edge_type] = []
                edge_type_list[edge_type].append((node_b_id, node_a_id))
        for edge_type in edge_type_list:
            edge_type_list[edge_type] = list(set(edge_type_list[edge_type]))
            edge_type_list_lim[edge_type] = len(edge_type_list[edge_type])

        self.edge_type_list = edge_type_list
        self.edge_type_list_lim = edge_type_list_lim


class DataTest:
    def __init__(self, Data_Sampler, batch_size):
        self.filenames = []
        self.Data_Sampler = Data_Sampler
        self.metapath_idx_key = list(self.Data_Sampler.metapath_to_edge_type.keys())
        self.batch_size = batch_size
        self.num_processed_edge = 0
        self.num_edge_all = Data_Sampler.num_edge
        self.ds = tf.data.Dataset.from_generator(
        self.data_gen,
        (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64))

    def data_gen(self):
        i = 0
        while True:
            key = self.metapath_idx_key[i]
            i += 1
            if i == len(self.metapath_idx_key):
                i = 0
            if self.num_processed_edge > self.num_edge_all:
                self.num_processed_edge = 0
                break
            yield np.array(self.Data_Sampler.training_pair(key, 1)[0],
                           dtype='int32')

    def data_read_in(self):
        dataset = tf.compat.v1.data.Dataset.from_generator(
            generator=self.data_gen,
            output_types=tf.int64,
            output_shapes=tf.TensorShape([None]))
        dataset = dataset.shuffle(400)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        return iterator, next_element







