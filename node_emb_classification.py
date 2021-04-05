import os, pickle, argparse, ast
import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,accuracy_score

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

def gen_label(config, target_node_type, target_file_name, filename = 'dblp'):
    data_info = pickle.load(open(filename+'_info_dict.p','rb'))
    data_index_to_id = pickle.load(open(filename+'_id_to_index.p', 'rb'))

    with open("{}_label.txt".format(filename), 'r') as f:
        paper_id_to_label = dict()
        paper_count = 0
        for line in f:
            paper_id, paper_label= line.rstrip().split(' ')
            paper_id_to_label[paper_id] = paper_label
            paper_count += 1

    node_start_index = data_info['off_set_min'][config['nodes'].index(target_node_type)]

    label = []
    labeled_index = []
    for k in range(len(data_index_to_id['out_mapping'][target_node_type])):
        node_id = data_index_to_id['out_mapping'][target_node_type][k]
        try:
            label.append(int(paper_id_to_label[node_id]))
            labeled_index.append(k+node_start_index)
        except:
            print('no label for {}:{}'.format(target_node_type, node_id))

    pickle.dump(
        {'label':label,
         'labeled_index':labeled_index
         }, open(target_file_name, 'wb'))

def my_svm(x, y, split_list=[0.2, 0.4, 0.6, 0.8], time=10, shuffle=True):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)
    y = np.squeeze(y)
    print('num_nodes: {}'.format(x.shape[0]))
    output = []
    for split in split_list:
        ss = split
        split = int(x.shape[0] * split)
        micro_list = []
        macro_list = []
        acc = []
        weight = []
        if time:
            for i in range(time):
                if shuffle:
                    permutation = np.random.permutation(x.shape[0])
                    x = x[permutation, :]
                    y = y[permutation]
                train_x = x[:split, :]
                test_x = x[split:, :]

                train_y = y[:split]
                test_y = y[split:]

                estimators = []
                estimators.append(svm.LinearSVC(dual=False))

                clf = make_pipeline(*estimators)
                clf.fit(train_x, train_y)
                y_pred = clf.predict(test_x)

                acc_s = accuracy_score(test_y,y_pred)
                f1_macro = f1_score(test_y, y_pred, average='macro')
                f1_micro = f1_score(test_y, y_pred, average='micro')
                f1_weight = f1_score(test_y, y_pred, average='weighted')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)
                acc.append(acc_s)
                weight.append(f1_weight)

            print('SVM_test(avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}, acc: {:.4f}, weighted: {:.4f}'.format(
                ss, time, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list), sum(acc)/ len(acc), sum(weight)/len(weight)))
            output.append(['{:.4f}'.format(sum(macro_list) / len(macro_list)),
                           '{:.4f}'.format(sum(micro_list) / len(micro_list)),
                           '{:.4f}'.format(sum(weight) / len(weight))])
    return output

def parse_args():
    '''
    Parses the mSHINE arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Evaluation.")

    parser.add_argument('--dimensions', type=int, default= 128, help='Number of dimensions. Default is 128.')

    parser.add_argument('--experi_dir', type=str, default='experi_data', help='Batch size. Default is 30.')

    parser.add_argument('--target_node_type', type=str, default='p', help='Record iter. Default is 50')

    parser.add_argument('--graph_name', type=str, default='cora', help='Name of dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ## PATH DEFINE
    experi_data_folder = os.path.join('data',args.graph_name, args.experi_dir)
    trans_file_dir = os.path.join(experi_data_folder,'record')

    iters = []
    for file_name in os.listdir(trans_file_dir):
        if '_'.join(file_name.split('_')[:2]) == 'trans_metric':
            iters.append(int(file_name.split('_')[-2]))

    iter_max = max(iters)

    ## FILE LOADED
    config_file = os.path.join('data/{}/{}.config'.format(args.graph_name, args.graph_name))
    emb = pickle.load(open('{}/node_emb/{}_{}_{}.emb'.format(experi_data_folder, args.graph_name, iter_max, args.dimensions),'rb'))
    trans_metric = pickle.load(open('{}/trans_metric_{}_{}'.format(trans_file_dir,iter_max, args.dimensions),'rb'))

    edge_type_metric = np.squeeze(trans_metric['edge_type_metric'])
    node_to_metapath_metric_x = np.squeeze(trans_metric['metapath_type_metric_x'])
    node_to_metapath_metric_h = np.squeeze(trans_metric['metapath_type_metric_h'])
    node_to_metapath_metric_y = np.squeeze(trans_metric['metapath_type_metric_y'])
    num_metapath = np.shape(node_to_metapath_metric_x)[0]

    print('max iter:{}'.format(iter_max))
    config = read_config(config_file)
    key_node_type_idx = config['nodes'].index(args.target_node_type)
    target_metapath_index = config['node_to_metapath_indicator'][key_node_type_idx]

    ## LABEL GENERATION
    label_file = 'data/{}/{}_TEST.label'.format(args.graph_name, args.graph_name)
    if os.path.isfile(label_file):
        label_file = pickle.load(open(label_file, 'rb'))
    else:
        gen_label(config, args.target_node_type, label_file, 'data/{}/{}'.format(args.graph_name, args.graph_name))
        label_file = pickle.load(open(label_file, 'rb'))

    label = label_file['label']
    labeled_index = np.array(label_file['labeled_index'])

    print(labeled_index.shape)
    s_embedding = emb['s_embedding'][labeled_index]
    i_embedding = emb['i_embedding'][labeled_index]
    o_embedding = emb['o_embedding'][labeled_index]

    for metapath_type_index in range(num_metapath):
        filter_x = np.reshape(node_to_metapath_metric_x[metapath_type_index, :], newshape=[1, -1])
        filter_h = np.reshape(node_to_metapath_metric_h[metapath_type_index, :], newshape=[1, -1])
        filter_y = np.reshape(node_to_metapath_metric_y[metapath_type_index, :], newshape=[1, -1])

        i_embedding_emb = i_embedding * filter_x
        s_embedding_emb = s_embedding * filter_h
        o_embedding_emb = o_embedding * filter_y
        print(config['metapath_types'][metapath_type_index])
        print('\tinput')
        my_svm(i_embedding_emb, label, time=10)
