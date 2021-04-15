# mSHINE
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FXinyiZ001%2FmSHINE&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

The official Tensorflow implementation of mSHINE: A Multiple-meta-paths Simultaneous Learning Framework for Heterogeneous Information Network Embedding. [mSHINE](https://ieeexplore.ieee.org/document/9201301).

*** 

### Package Version

     Keras-Preprocessing==1.1.2
     numpy==1.19.2
     scikit-learn==0.24.1
     scipy==1.4.1
     tensorflow-gpu==2.2.0
    
***
## Data Preparation

The experimental datasets used in paper are available at https://drive.google.com/file/d/1g3Ln0fzCIqUO7A1GTZpKntXSUaiJdqgZ/view?usp=sharing

To run mSHINE on your HIN whose name is **XXX**(we use the dataset cora as an example blow), two files should be provided: 
   1)**XXX.hin** (HIN dataset); 
   2)**XXX.config** (Meta-path info)

1) The supported input HIN format is an edgelist (separated by space):
    
    ```
     node_type_1:node_id_1 node_type_2:node_id_2 edge_weights node_type_1:node_type_2
     ...
    ```
2) Generate **XXX.config**:
   
   - **XXX.relation** file which lists all the possible egde types should be provided. The format of **XXX.relation**:
   ```
    node_type_1-node_type_2
    node_type_2-node_type_1
    node_type_1-node_type_3
    ...
   ```
   (*NOTE*: The edge is assumed to be directed by default, you need to use two directed edges to represent an undirected edge type.)

   - To generate **XXX.config**:
   ```
    python config_file_gen/metapath_gen.py --dn cora --output_dir config_file_gen/config_files/ --relation_dir config_file_gen/relation_files/
   ```
3) Generate other necessary dataset files(i.e. **XXX.HIN_dic**, **XXX_id_to_index.p** and **XXX_info_dict.p**):

   - Be sure **XXX.hin** and **XXX.config** are put in the folder: *data/XXX/*, then:
   ```markdown
    cd data_prepare/
    python data_prepare.py --input_data cora
   ```
   The resulted files can be found in *data/XXX/*

***
## Excute

To run mSHINE:

   ```
    python main.py --graph_name cora --dimensions 128 --batch_size 128 --iter 1000
   ```
***
## Output

- The node representaions **XXX\_{epoch}\_{emb_size}.emb** can be found in: *data/XXX/experi_data/node_emb/*
   
   the embeddings are stored in the form of dictionary in a pickle file and can be loaded:
  ```
   import pickle
     
   emb = pickle.load(open('{path_of_emb_file}'),'rb'))
   state_embedding = emb['s_embedding']
   input_embedding = emb['i_embedding']
   output_embedding = emb['o_embedding']
  ```
  (*NOTE*: the mapping from index of node in state_embedding matrix to the node id can be found in **XXX_id_to_index.p** )
- The transform matrix stored in **trans_metric_{epoch}_{emb_size}** can be found in: *data/XXX/experi_data/record/*

***
## Evaluation

- An example of evalution code is available in **node_emb_classification.py** where a label file **XXX_label.txt** is required.
 
  The format of **XXX_label.txt** is:
    ```
     node_id_1 node_label
     node_id_2 node_label
     ...
    ```
  
  To run the evaluation:
  ```
    python node_emb_classification.py --dimensions 128 --experi_dir experi_data --target_node_type p --graph_name cora
  ```
  during which, **XXX_TEST.label** is generated for storing classification related info.

***
## Citing

If you find mSHINE is useful for your research, please consider citing the following paper:

    @ARTICLE{9201301,  
        author={X. {Zhang} and L. {Chen}},  
        journal={IEEE Transactions on Knowledge and Data Engineering},   
        title={mSHINE: A Multiple-meta-paths Simultaneous Learning Framework for Heterogeneous Information Network Embedding},   
        year={2020},
        volume={},
        number={},
        pages={1-1},
        doi={10.1109/TKDE.2020.3025464}}

Please send any questions you might have about the codes and/or the algorithm to xinyi001@e.ntu.edu.sg.
