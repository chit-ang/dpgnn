from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

#数据设置
class NodeMinibatchIterator(object):
    def __init__(self, graphs, features, adjs, placeholders, num_time_steps, context_pairs=None,batch_size=100):
        self.graphs = graphs
        self.features = features
        self.adjs = adjs
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.batch_num = 0
        self.num_time_steps = num_time_steps
        self.degs = self.construct_degs()
        self.context_pairs = context_pairs
        self.max_positive = FLAGS.neg_sample_size
        self.train_nodes = self.graphs[num_time_steps-1].nodes() # all nodes in the graph.
        print ("# train nodes", len(self.train_nodes))

    #计算节点度数
    def construct_degs(self):
        degs = []
        for i in range(0, self.num_time_steps):
            G = self.graphs[i]
            deg = np.zeros((len(G.nodes()),))
            for nodeid in G.nodes():
                neighbors = np.array(list(G.neighbors(nodeid)))
                deg[nodeid] = len(neighbors)
            degs.append(deg)
        min_t = 0
        if FLAGS.window > 0:
            min_t = max(self.num_time_steps - FLAGS.window - 1, 0)
        return degs[min_t:]

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    #依据随机游走结果获取上下文节点，正负样本
    def batch_feed_dict(self, batch_nodes):
        node_1_all = []
        node_2_all = []
        min_t = 0
        if FLAGS.window > 0:
            min_t = max(self.num_time_steps - FLAGS.window - 1, 0)
        for t in range(min_t, self.num_time_steps):
            node_1 = []
            node_2 = []

            for n in batch_nodes:
                if len(self.context_pairs[t][n]) > self.max_positive:
                    node_1.extend([n]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][n], self.max_positive, replace=False))
                else:
                    node_1.extend([n]* len(self.context_pairs[t][n]))
                    node_2.extend(self.context_pairs[t][n])

            assert len(node_1) == len(node_2)
            assert len(node_1) <= self.batch_size * self.max_positive
            feed_dict = dict()
            node_1_all.append(node_1)
            node_2_all.append(node_2)

        feed_dict.update({self.placeholders['node_1'][t-min_t]:node_1_all[t-min_t] for t in range(min_t, self.num_time_steps)})
        feed_dict.update({self.placeholders['node_2'][t-min_t]:node_2_all[t-min_t] for t in range(min_t, self.num_time_steps)})
        feed_dict.update({self.placeholders['features'][t-min_t]:self.features[t] for t in range(min_t, self.num_time_steps)})
        feed_dict.update({self.placeholders['adjs'][t-min_t]: self.adjs[t] for t in range(min_t, self.num_time_steps)})
        feed_dict.update({self.placeholders['batch_nodes']:np.array(batch_nodes).astype(np.int32)})
        return feed_dict

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def shuffle(self):
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        self.train_nodes =  self.graphs[self.num_time_steps-1].nodes()
        self.batch_num = 0
