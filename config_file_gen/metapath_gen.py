import os,argparse

def parse_args():

    parser = argparse.ArgumentParser(description="metapath config file gen")

    parser.add_argument('--dn', type=str, default='movie_fake', help='The name of dataset')

    parser.add_argument('--output_dir', type=str, default='config_files', help='The output dir of generated files')

    parser.add_argument('--relation_dir', type=str, default='relation_files', help='The dir of input relation files')

    return parser.parse_args()

def find_edges(current_edge, current_metapath_list, metapath_list,current_metapath_str, metapath_str):
    for third_node in edge_dict[current_edge]:
        new_edge = current_edge[1:] + third_node
        if new_edge in current_metapath_list:
            repeat_idx = current_metapath_list.index(new_edge)
            current_metapath_list_ = current_metapath_list[repeat_idx:]
            current_metapath_str_ = current_metapath_str[repeat_idx:]
            metapath_list.append(list(current_metapath_list_))
            metapath_str.append(current_metapath_str_)
        elif new_edge not in current_metapath_list:
            current_metapath_list.append(new_edge)
            current_metapath_str += third_node
            current_metapath_list, current_metapath_str, metapath_list, metapath_str = find_edges(new_edge,
                                                                                                  current_metapath_list,
                                                                                                  metapath_list,
                                                                                                  current_metapath_str,
                                                                                                  metapath_str)
    del current_metapath_list[-1]
    current_metapath_str = current_metapath_str[:-1]
    return current_metapath_list, current_metapath_str, metapath_list, metapath_str


def metapath_gen(metapath_list):
    metapath_list_out = []
    for metapath in metapath_list:
        metapath_out = []
        for edge in metapath:
            edge_out = ':'.join(list(edge))
            metapath_out.append('\''+edge_out+'\'')
        metapath_list_out.append('['+','.join(metapath_out)+']')
    str_out = '['+','.join(metapath_list_out)+']'
    return str_out


def three_elem_edge_gen(metapath_str):
    metapath_list = []
    key_edge_list = []
    for metapath_str_i in metapath_str:
        edge_list = edge_to_edge_list(metapath_str_i)
        metapath_list.append(edge_list)
        key_edge_list.append(edge_list[-1][:2])
    return metapath_list, key_edge_list


def node_list_gen(metapath_list):
    node_list = []
    node_to_metapath_indicator = []
    for node_type in node_connection_dict:
        node_list.append(node_type)

    for node_type in node_list:
        node_to_metapath_indicator_i = []
        for metapath_id,metapath in enumerate(metapath_list):
            if_exist = -1
            for edge in metapath:
                if node_type in edge:
                    if_exist = 1
            if if_exist !=-1:
                node_to_metapath_indicator_i.append(str(metapath_id))
        node_to_metapath_indicator.append(node_to_metapath_indicator_i)

    node_list = ['\''+node+'\'' for node in node_list]
    str_out_node = '[' + ','.join(node_list) + ']'

    metapath_list_out = []
    for metapath in node_to_metapath_indicator:
        edge_out = ','.join(metapath)
        metapath_list_out.append('[' + edge_out + ']')
    str_out_indicator = '[' + ','.join(metapath_list_out) + ']'

    return str_out_node, str_out_indicator


def edge_list_gen(metapath_list):
    edge_list = set()
    for metapath in metapath_list:
        for edge in metapath:
            edge_list.add('\''+':'.join(list(edge))+'\'')
    edge_list = list(edge_list)
    str_out_edge = '[' + ','.join(edge_list) + ']'
    return str_out_edge

def edge_to_edge_list(node_list):
    node_list = node_list + node_list[1]
    edge_list = []
    for idx in range(len(node_list)-2):
        edge_list.append(node_list[idx:idx+3])
    return edge_list

def symmetric_edge_select(metapath_list):
    new_metapath_list = []
    for metapath in metapath_list:
        metapath_string = ''
        for node_index, node in enumerate(metapath):
            if node_index == 0:
                metapath_string = node
            else:
                metapath_string += node[-1]
        metapath_string = metapath_string[:-1]
        metapath_string_reverse = metapath_string[::-1]
        if metapath_string == metapath_string_reverse:
            new_metapath_list.append(metapath)
    return new_metapath_list


if __name__ == '__main__':

    args = parse_args()
    input_file_path = os.path.join(args.relation_dir,args.dn+'.relation')

    node_connection_dict = dict()
    ## build connection dict
    with open(input_file_path,'r') as f:
        for line in f.readlines():
            node_a,node_b = line.rstrip().split('-')
            if not node_a in node_connection_dict:
                node_connection_dict[node_a] = []
            node_connection_dict[node_a].append(node_b)

    ## build edge dict
    edge_dict = dict()
    for node_a in node_connection_dict:
        for node_b in node_connection_dict[node_a]:
            for node_c in node_connection_dict[node_b]:
                if not node_a+node_b in edge_dict:
                    edge_dict[node_a+node_b] = []
                edge_dict[node_a+node_b].append(node_c)



    metapath_list_all = []
    metapath_str_all = []
    for node_a in node_connection_dict:
        for node_b in node_connection_dict[node_a]:
            current_edge = node_a + node_b
            current_metapath_list, current_metapath_str, metapath_list, metapath_str = find_edges(current_edge,
                                                                                                  [current_edge],
                                                                                                  [],
                                                                                                  current_edge,
                                                                                                  [])

            metapath_list_all.extend(list(metapath_list))
            metapath_str_all.extend(list(metapath_str))

    metapath_list_all, key_edge_list = three_elem_edge_gen(metapath_str_all)
    metapath_list_all = symmetric_edge_select(metapath_list_all)
    metapath_list = [tuple(sorted(list(set(metapath)))) for metapath in metapath_list_all]
    metapath_list_all = list(set(metapath_list))
    node_line,node_to_metapath_line = node_list_gen(metapath_list_all)
    edge_line = edge_list_gen(metapath_list_all)
    metapath_line = metapath_gen(metapath_list_all)

    with open(os.path.join(args.output_dir,args.dn+'.config'),'w') as f:
        f.write(node_line+'\n')
        f.write(edge_line+'\n')
        f.write(metapath_line+'\n')
        f.write(node_to_metapath_line+'\n')

