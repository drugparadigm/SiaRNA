import sys
import os

from collections import defaultdict
import numpy as np

class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_unseen_dataSet(self, dataSet='NPInter5', adj_lists=None, node_map=None, feat_data=None):
        # print(self.config)
        feat_file = dataSet + '_graphsage_dataset/' + dataSet + '_feats' + '.txt'
        # print(feat_file)
        # feat_data = np.asarray(feat_data)
        # print(f'feat_data in dataCenter1:\n {feat_data}')
        # print(f'feat_data.shape in dataCenter1:\n {feat_data.shape}')
        # interaction_file = self.config['file_path.' + dataSet + '_interaction']  # 所有positive的interactions
        adj_lists_keys = adj_lists.keys()

        max_node_index = max(adj_lists_keys)  # 原本数据集中节点数量
        # print(f"max_node_index: {max_node_index}")


        with open(feat_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                # feat_data.append([float(x) for x in info[1:-1]])
                feat_data.append([float(x) for x in info[1:-1]])
                node_map[info[0]] = i + max_node_index
                # feat_map[info[0]] = i
                # if not info[-1] in label_map:
                #     label_map[info[-1]] = len(label_map)
                # labels.append(label_map[info[-1]])

        length_feat_data = len(feat_data)

        feat_data = np.asarray(feat_data)
        # print(f'feat_data in dataCenter:\n {feat_data}')
        # print(f'feat_data.shape in dataCenter1:\n {feat_data.shape}')
        # print(feat_data)
        unseen_indexs = np.empty(shape=(0,))

        for idx in range(max_node_index+1, length_feat_data):
            unseen_indexs = np.append(unseen_indexs, idx)
        unseen_indexs = unseen_indexs.astype(int)

        setattr(self, dataSet + '_unseen', unseen_indexs)


        return unseen_indexs, feat_data




    def load_dataSet(self, dataSet='RPI2241', embedding_type=None):
        # if dataSet == 'cora':
        #     cora_content_file = self.config['file_path.cora_content']
        #     cora_cite_file = self.config['file_path.cora_cite']
        #
        #     feat_data = []
        #     # labels = []  # label sequence of node
        #     node_map = {}  # map node to Node_ID
        #     # label_map = {}  # map label to Label_ID
        #     feat_map = {}  # map feat to Node_ID
        #     with open(cora_content_file) as fp:
        #         for i, line in enumerate(fp):
        #             info = line.strip().split()
        #             feat_data.append([float(x) for x in info[1:-1]])
        #             node_map[info[0]] = i
        #             feat_map[info[0]] = i
        #             # if not info[-1] in label_map:
        #             #     label_map[info[-1]] = len(label_map)
        #             # labels.append(label_map[info[-1]])
        #     feat_data = np.asarray(feat_data)
        #
        #     # labels = np.asarray(labels, dtype=np.int64)
        #
        #     #获取边列表
        #     edges_lists = []
        #     with open(cora_cite_file) as fp:
        #         for i, line in enumerate(fp):
        #             node1, node2 = line.strip().split()
        #             edges_lists.append((node1, node2))
        #
        #     # test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
        #     test_indexs, val_indexs, train_indexs = self._split_data_edge(len(edges_lists), node_map, edges_lists)
        #
        #     setattr(self, dataSet + '_test', test_indexs)
        #     setattr(self, dataSet + '_val', val_indexs)
        #     setattr(self, dataSet + '_train', train_indexs)
        #
        #     adj_lists = defaultdict(set)
        #     with open(cora_cite_file) as fp:
        #         for i, line in enumerate(fp):
        #             info = line.strip().split()
        #             assert len(info) == 2
        #             paper1 = node_map[info[0]]
        #             paper2 = node_map[info[1]]
        #             if (((paper1 in train_indexs) and (paper2 in train_indexs))) or (((paper1 in val_indexs) and (paper2 in val_indexs))):
        #                 adj_lists[paper1].add(paper2)
        #                 adj_lists[paper2].add(paper1)
        #     #print(adj_lists)
        #     #print(type(adj_lists))
        #
        #         #print(edges_lists)
        #
        #     print(len(train_indexs))
        #     print(len(val_indexs))
        #     print(len(test_indexs))
        #     # print(adj_lists)
        #
        #     # assert len(feat_data) == len(labels) == len(adj_lists)
        #
        #     setattr(self, dataSet + '_feats', feat_data)
        #     # setattr(self, dataSet + '_labels', labels)
        #     setattr(self, dataSet + '_adj_lists', adj_lists)
        #
        # elif dataSet == 'RPI2241':
        #     RPI2241_feat_file = self.config['file_path.RPI2241_feat']
        #     RPI2241_interaction_file = self.config['file_path.RPI2241_interaction']
        #
        #     feat_data = []
        #     # labels = []  # label sequence of node
        #     node_map = {}  # map node to Node_ID
        #     # label_map = {}  # map label to Label_ID
        #     feat_map = {}  # map feat to Node_ID
        #     with open(RPI2241_feat_file) as fp:
        #         for i, line in enumerate(fp):
        #             info = line.strip().split()
        #             feat_data.append([float(x) for x in info[1:-1]])
        #             node_map[info[0]] = i
        #             feat_map[info[0]] = i
        #             # if not info[-1] in label_map:
        #             #     label_map[info[-1]] = len(label_map)
        #             # labels.append(label_map[info[-1]])
        #     feat_data = np.asarray(feat_data)
        #
        #     # labels = np.asarray(labels, dtype=np.int64)
        #     # print(node_map)
        #     #获取边列表
        #     edges_lists = []
        #     with open(RPI2241_interaction_file) as fp:
        #         for i, line in enumerate(fp):
        #             node1, node2 = line.strip().split()
        #             edges_lists.append((node1, node2))
        #
        #     # test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
        #     test_indexs, val_indexs, train_indexs = self._split_data_edge(len(edges_lists), node_map, edges_lists)
        #
        #     setattr(self, dataSet + '_test', test_indexs)
        #     setattr(self, dataSet + '_val', val_indexs)
        #     setattr(self, dataSet + '_train', train_indexs)
        #
        #     adj_lists = defaultdict(set)
        #     with open(RPI2241_interaction_file) as fp:
        #         for i, line in enumerate(fp):
        #             info = line.strip().split()
        #             assert len(info) == 2
        #             paper1 = node_map[info[0]]
        #             paper2 = node_map[info[1]]
        #             if (((paper1 in train_indexs) and (paper2 in train_indexs))) or (((paper1 in val_indexs) and (paper2 in val_indexs))):
        #                 adj_lists[paper1].add(paper2)
        #                 adj_lists[paper2].add(paper1)
        #     #print(adj_lists)
        #     #print(type(adj_lists))
        #
        #         #print(edges_lists)
        #
        #     print(len(train_indexs))
        #     print(len(val_indexs))
        #     print(len(test_indexs))
        #     # print(adj_lists)
        #
        #     # assert len(feat_data) == len(labels) == len(adj_lists)
        #
        #     setattr(self, dataSet + '_feats', feat_data)
        #     # setattr(self, dataSet + '_labels', labels)
        #     setattr(self, dataSet + '_adj_lists', adj_lists)
        #
        # elif dataSet == 'RPI2241_train':
        #     RPI2241_feat_file = self.config['file_path.RPI2241_feat']
        #     RPI2241_interaction_file = self.config['file_path.RPI2241_interaction'] #所有positive的interactions
        #     RPI2241_train_interaction_file = self.config['file_path.RPI2241_train_interaction'] #包含所有positive且在训练集中的interactions
        #     RPI2241_test_interaction_file = self.config['file_path.RPI2241_test_interaction'] #包含所有positive且在测试集中出现的interactions以及所有negative的样本
        #     feat_data = []
        #     # labels = []  # label sequence of node
        #     node_map = {}  # map node to Node_ID
        #     # label_map = {}  # map label to Label_ID
        #     feat_map = {}  # map feat to Node_ID
        #     with open(RPI2241_feat_file) as fp:
        #         for i, line in enumerate(fp):
        #             info = line.strip().split()
        #             feat_data.append([float(x) for x in info[1:-1]])
        #             node_map[info[0]] = i
        #             feat_map[info[0]] = i
        #             # if not info[-1] in label_map:
        #             #     label_map[info[-1]] = len(label_map)
        #             # labels.append(label_map[info[-1]])
        #     feat_data = np.asarray(feat_data)
        #
        #     # labels = np.asarray(labels, dtype=np.int64)
        #     # print(node_map)
        #     #获取边列表
        #     edges_lists = []
        #     with open(RPI2241_interaction_file) as fp:
        #         for i, line in enumerate(fp):
        #             node1, node2 = line.strip().split()
        #             edges_lists.append((node1, node2))
        #
        #     edges_lists_train = []
        #     with open(RPI2241_train_interaction_file) as fp:
        #         for i, line in enumerate(fp):
        #             node1, node2 = line.strip().split()
        #             edges_lists_train.append((node1, node2))
        #
        #     edges_lists_test = []
        #     with open(RPI2241_test_interaction_file) as fp:
        #         for i, line in enumerate(fp):
        #             node1, node2 = line.strip().split()
        #             edges_lists_test.append((node1, node2))
        #     # test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
        #
        #     # test_indexs, val_indexs, train_indexs = self._split_data_edge(len(edges_lists), node_map, edges_lists)
        #
        #     test_indexs, val_indexs, train_indexs = self._split_data_vecnet(edges_lists_train, edges_lists_test, node_map)
        #
        #     setattr(self, dataSet + '_test', test_indexs)
        #     setattr(self, dataSet + '_val', val_indexs)
        #     setattr(self, dataSet + '_train', train_indexs)
        #
        #     adj_lists = defaultdict(set)
        #     with open(RPI2241_interaction_file) as fp:
        #         for i, line in enumerate(fp):
        #             info = line.strip().split()
        #             assert len(info) == 2
        #             paper1 = node_map[info[0]]
        #             paper2 = node_map[info[1]]
        #             if (((paper1 in train_indexs) and (paper2 in train_indexs))) or (((paper1 in val_indexs) and (paper2 in val_indexs))):
        #                 adj_lists[paper1].add(paper2)
        #                 adj_lists[paper2].add(paper1)
        #     #print(adj_lists)
        #     #print(type(adj_lists))
        #
        #         #print(edges_lists)
        #
        #     print(len(train_indexs))
        #     print(len(val_indexs))
        #     print(len(test_indexs))
        #     # print(adj_lists)
        #
        #     # assert len(feat_data) == len(labels) == len(adj_lists)
        #
        #     setattr(self, dataSet + '_feats', feat_data)
        #     # setattr(self, dataSet + '_labels', labels)
        #     setattr(self, dataSet + '_adj_lists', adj_lists)
        # dataSet

        # feat_file = self.config['file_path.' + dataSet + '_feat']
        # interaction_file = self.config['file_path.' + dataSet + '_interaction']  # 所有positive的interactions
        # train_interaction_file = self.config['file_path.' + dataSet + '_train_interaction']  # 包含所有positive且在训练集中的interactions
        # test_interaction_file = self.config['file_path.' + dataSet + '_test_interaction']  # 包含所有positive且在测试集中出现的interactions以及所有negative的样本

        if embedding_type == None:
            feat_file =  dataSet + '_graphsage_dataset/'+ dataSet +'_feats.txt'
            interaction_file = dataSet + '_graphsage_dataset/'+ dataSet +'_total_interactions_seq_list.txt'
            train_interaction_file = dataSet + '_graphsage_dataset/'+ dataSet +'_graphsage_train_interactions.txt'
            test_interaction_file = dataSet + '_graphsage_dataset/'+ dataSet +'_graphsage_test_interactions.txt'
        else:
            feat_file = dataSet + '_' + embedding_type + '_graphsage_dataset/' + dataSet +'_feats.txt'
            interaction_file = dataSet + '_' + embedding_type + '_graphsage_dataset/' + dataSet +'_total_interactions_seq_list.txt'
            train_interaction_file = dataSet + '_' + embedding_type + '_graphsage_dataset/' + dataSet +'_graphsage_train_interactions.txt'
            test_interaction_file = dataSet + '_' + embedding_type + '_graphsage_dataset/' + dataSet +'_graphsage_test_interactions.txt'

        feat_data = []
        # labels = []  # label sequence of node
        node_map = {}  # map node to Node_ID

        # label_map = {}  # map label to Label_ID
        feat_map = {}  # map feat to Node_ID
        with open(feat_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                # feat_data.append([float(x) for x in info[1:-1]])
                feat_data.append([float(x) for x in info[1:-1]])
                node_map[info[0]] = i
                feat_map[info[0]] = i
                # if not info[-1] in label_map:
                #     label_map[info[-1]] = len(label_map)
                # labels.append(label_map[info[-1]])
        feat_data = np.asarray(feat_data)
        # print(info[0])
        # print(f"node_map: {node_map[info[0]]}")
        # print(f'feat_data in dataCenter1:\n {feat_data}')
        # print(f'feat_data.shape in dataCenter1:\n {feat_data.shape}')
        # labels = np.asarray(labels, dtype=np.int64)
        # print(node_map)
        # 获取边列表
        edges_lists = []
        with open(interaction_file) as fp:
            for i, line in enumerate(fp):
                node1, node2 = line.strip().split()
                edges_lists.append((node1, node2))

        edges_lists_train = []
        with open(train_interaction_file) as fp:
            for i, line in enumerate(fp):
                node1, node2 = line.strip().split()
                edges_lists_train.append((node1, node2))

        edges_lists_test = []
        with open(test_interaction_file) as fp:
            for i, line in enumerate(fp):
                node1, node2 = line.strip().split()
                edges_lists_test.append((node1, node2))
        # test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

        # test_indexs, val_indexs, train_indexs = self._split_data_edge(len(edges_lists), node_map, edges_lists)

        test_indexs, val_indexs, train_indexs = self._split_data_vecnet(edges_lists_train, edges_lists_test, node_map)

        setattr(self, dataSet + '_test', test_indexs)
        setattr(self, dataSet + '_val', val_indexs)
        setattr(self, dataSet + '_train', train_indexs)

        adj_lists = defaultdict(set)
        with open(interaction_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                assert len(info) == 2
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                if (((paper1 in train_indexs) and (paper2 in train_indexs))) or (
                ((paper1 in val_indexs) and (paper2 in val_indexs))):
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)
        # print(adj_lists)
        # print(type(adj_lists))

        # print(edges_lists)

        # print(len(train_indexs))
        # print(len(val_indexs))
        # print(len(test_indexs))
        # print(adj_lists)

        # assert len(feat_data) == len(labels) == len(adj_lists)

        setattr(self, dataSet + '_feats', feat_data)
        # setattr(self, dataSet + '_labels', labels)
        setattr(self, dataSet + '_adj_lists', adj_lists)

        return node_map, feat_data

    def _split_data(self, num_nodes, test_split=3, val_split=6):
        rand_indices = np.random.permutation(num_nodes)

        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size + val_size)]
        train_indexs = rand_indices[(test_size + val_size):]

        return test_indexs, val_indexs, train_indexs

    def _split_data_edge(self, num_edges, node_map, edges_lists, test_split=5, val_split=10):
        rand_indices = np.random.permutation(num_edges)
        ### 通过边列表来划分训练集，测试集，和验证集
        test_edge_size = num_edges // test_split
        val_edge_size = num_edges // val_split
        train_edge_size = num_edges - (test_edge_size + val_edge_size)

        test_edge_indexs = rand_indices[:test_edge_size]
        val_edge_indexs = rand_indices[test_edge_size:(test_edge_size + val_edge_size)]
        train_edge_indexs = rand_indices[(test_edge_size + val_edge_size):]

        #print(test_edge_indexs)

        edges_array = np.array(edges_lists)

        test_edge = edges_array[test_edge_indexs]
        val_edge = edges_array[val_edge_indexs]
        train_edge = edges_array[train_edge_indexs]

        test_indexs  = np.empty(shape=(0,))
        val_indexs = np.empty(shape=(0,))
        train_indexs = np.empty(shape=(0,))

        #print(test_edge)
        for idx in range(len(test_edge)):
            edges_idx = test_edge[idx]
            # node1 = int(edges_idx[0])
            # node2 = int(edges_idx[1])
            node1 = edges_idx[0]
            node2 = edges_idx[1]
            node1_index = node_map[str(node1)]
            node2_index = node_map[str(node2)]
            test_indexs = np.append(test_indexs, int(node1_index))
            test_indexs = np.append(test_indexs, int(node2_index))
        for idx in range(len(val_edge)):
            edges_idx = val_edge[idx]
            # node1 = int(edges_idx[0])
            # node2 = int(edges_idx[1])
            node1 = edges_idx[0]
            node2 = edges_idx[1]
            node1_index = node_map[str(node1)]
            node2_index = node_map[str(node2)]
            val_indexs = np.append(val_indexs, int(node1_index))
            val_indexs = np.append(val_indexs, int(node2_index))

        for idx in range(len(train_edge)):
            edges_idx = train_edge[idx]
            # node1 = int(edges_idx[0])
            # node2 = int(edges_idx[1])
            node1 = edges_idx[0]
            node2 = edges_idx[1]
            node1_index = node_map[str(node1)]
            node2_index = node_map[str(node2)]
            train_indexs = np.append(train_indexs, int(node1_index))
            train_indexs = np.append(train_indexs, int(node2_index))

        # 将浮点数数组转换为整数数组
        test_indexs = test_indexs.astype(int)
        val_indexs = val_indexs.astype(int)
        train_indexs = train_indexs.astype(int)

        test_indexs = np.unique(test_indexs)
        val_indexs = np.unique(val_indexs)
        train_indexs = np.unique(train_indexs)

        return test_indexs, val_indexs, train_indexs

    def _split_data_vecnet(self, edges_lists_train, edges_lists_test, node_map, val_split=10):
        val_edge_size = len(edges_lists_train) // val_split
        edges_train_array = np.array(edges_lists_train)
        edges_test_array = np.array(edges_lists_test)

        train_edge = edges_train_array[val_edge_size:]
        val_edge = edges_train_array[:val_edge_size]
        test_edge = edges_test_array

        test_indexs  = np.empty(shape=(0,))
        val_indexs = np.empty(shape=(0,))
        train_indexs = np.empty(shape=(0,))

        for idx in range(len(test_edge)):
            edges_idx = test_edge[idx]
            # node1 = int(edges_idx[0])
            # node2 = int(edges_idx[1])
            node1 = edges_idx[0]
            node2 = edges_idx[1]
            node1_index = node_map[str(node1)]
            node2_index = node_map[str(node2)]
            test_indexs = np.append(test_indexs, int(node1_index))
            test_indexs = np.append(test_indexs, int(node2_index))
        for idx in range(len(val_edge)):
            edges_idx = val_edge[idx]
            # node1 = int(edges_idx[0])
            # node2 = int(edges_idx[1])
            node1 = edges_idx[0]
            node2 = edges_idx[1]
            node1_index = node_map[str(node1)]
            node2_index = node_map[str(node2)]
            val_indexs = np.append(val_indexs, int(node1_index))
            val_indexs = np.append(val_indexs, int(node2_index))

        for idx in range(len(train_edge)):
            edges_idx = train_edge[idx]
            # node1 = int(edges_idx[0])
            # node2 = int(edges_idx[1])
            node1 = edges_idx[0]
            node2 = edges_idx[1]
            node1_index = node_map[str(node1)]
            node2_index = node_map[str(node2)]
            train_indexs = np.append(train_indexs, int(node1_index))
            train_indexs = np.append(train_indexs, int(node2_index))

        # 将浮点数数组转换为整数数组
        test_indexs = test_indexs.astype(int)
        val_indexs = val_indexs.astype(int)
        train_indexs = train_indexs.astype(int)

        test_indexs = np.unique(test_indexs)
        val_indexs = np.unique(val_indexs)
        train_indexs = np.unique(train_indexs)
        # print(f"test_indexs: {test_indexs}")
        # print(f"val_indexs: {val_indexs}")
        # print(f"train_indexs: {train_indexs}")
        return test_indexs, val_indexs, train_indexs








