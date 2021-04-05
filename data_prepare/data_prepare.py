#################################################
## refers to https://github.com/GentleZhu/HEER ##
#################################################

import pickle,argparse, os, ast
import numpy as np


class HinLoader(object):
    """docstring for HinLoader"""

    def __init__(self, arg):
        self.in_mapping = dict()
        self.out_mapping = dict()
        self.HIN_dict = dict()
        self.input = list()
        self.output = list()
        self.edge_type = list()
        self.metapath_type = list()
        self.direction = list()
        self.arg = arg
        self.link_to_num = dict()
        # print(arg['types'])
        for k in arg['nodes']:
            self.in_mapping[k] = dict()
            self.out_mapping[k] = dict()
            self.HIN_dict[k] = dict()

    def inNodeMapping(self, key, type):
        if key not in self.in_mapping[type]:
            self.out_mapping[type][len(self.in_mapping[type])] = key
            self.in_mapping[type][key] = len(self.in_mapping[type])
        return self.in_mapping[type][key]

    def build_HIN(self, node_a, node_a_type, node_b, node_b_type, direction = '1'):
        if node_a not in self.HIN_dict[node_a_type]:
            self.HIN_dict[node_a_type][node_a] = dict()
        if node_b_type not in self.HIN_dict[node_a_type][node_a]:
            self.HIN_dict[node_a_type][node_a][node_b_type] = []

        self.HIN_dict[node_a_type][node_a][node_b_type].append(node_b)
        link = ':'.join([node_a_type,node_b_type])
        if link not in self.link_to_num:
            self.link_to_num[link] = 0
        self.link_to_num[link] += 1

        if direction == '1':
            if node_b not in self.HIN_dict[node_b_type]:
                self.HIN_dict[node_b_type][node_b] = dict()
            if node_a_type not in self.HIN_dict[node_b_type][node_b]:
                self.HIN_dict[node_b_type][node_b][node_a_type] = []
            self.HIN_dict[node_b_type][node_b][node_a_type].append(node_a)
            link = ':'.join([node_b_type, node_a_type])
            if link not in self.link_to_num:
                self.link_to_num[link] = 0
            self.link_to_num[link] += 1

    def readHin(self, _edge_types):
        print(' Hin readin')
        with open(self.arg['graph']) as INPUT:
            for line in INPUT:
                edge = line.strip().split(' ')
                node_a = edge[0].split(':')
                node_b = edge[1].split(':')
                self.build_HIN(node_a[1],node_a[0],node_b[1],node_b[0])
                self.inNodeMapping(node_a[1], node_a[0])
                self.inNodeMapping(node_b[1], node_b[0])

    def dump(self, dump_path):
        offset = 0
        self.encoder = dict()
        self.off_set_min = []
        self.num_node_each_type = []
        for k in self.arg['nodes']:
            self.encoder[k] = offset
            self.off_set_min.append(offset)
            self.num_node_each_type.append(len(self.in_mapping[k]))
            offset += len(self.in_mapping[k])
        self.encoder['sum'] = offset
        self.num_node = offset
        info_dict = dict()
        info_dict['encoder'] = self.encoder
        info_dict['num_node'] = self.num_node
        info_dict['off_set_min'] = np.array(self.off_set_min)
        info_dict['num_node_each_type'] = np.array(self.num_node_each_type)
        print(info_dict)
        pickle.dump(info_dict, open(dump_path + '_info_dict.p', 'wb'))
        pickle.dump(
            {'out_mapping':self.out_mapping,
             'in_mapping':self.in_mapping},
            open(dump_path+'_id_to_index.p','wb'))
        pickle.dump(self.HIN_dict,open(dump_path+'.HIN_dict','wb'))

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

def parse_args():

    parser = argparse.ArgumentParser(description="data_preprocessing")

    parser.add_argument('--input_data', type=str, default='cora', help='The name of dataset')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    data_name = args.input_data
    output_dir = os.path.join('../data',data_name)

    config_name = os.path.join(output_dir, data_name+ '.config')
    data_path = os.path.join(output_dir,data_name+'.hin')

    config = read_config(config_name)

    tmp = HinLoader({
        'graph': data_path,
        'nodes': config['nodes'],
        'edge_types':config['edge_types'],
        'node_to_metapath_indicator': config['node_to_metapath_indicator'],
        'metapath_types':config['metapath_types'],
        'metapath_dict':config['metapath_dict']})
    tmp.readHin(config['edge_types'])
    tmp.dump(os.path.join(output_dir,data_name))
