#节点训练+预测训练（干扰数据+干扰标签）LP
from __future__ import division
from __future__ import print_function

import json
import os
import time
from datetime import datetime

import logging
import scipy
import numpy as np
from task.link_prediction import  write_to_csv,evaluate_classifier
from flags import *
from models.models import DG
from utils.minibatch import *
from utils.preprocess import *
from utils.utilities import *

np.random.seed(123)
tf.compat.v1.set_random_seed(123)
flags = tf.app.flags
FLAGS = flags.FLAGS
output_dir = "./logs/"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

LOG_DIR = output_dir + FLAGS.log_dir
SAVE_DIR = output_dir + FLAGS.save_dir
CSV_DIR = output_dir + FLAGS.csv_dir
MODEL_DIR = output_dir + FLAGS.model_dir

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU_ID)
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()
log_file = LOG_DIR + '/%s_%s_%s_%s_%s.log' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day), str(FLAGS.time_steps))
log_level = logging.INFO
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logging.info(FLAGS.flag_values_dict().items())
output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

num_time_steps = FLAGS.time_steps
epsilon_1=FLAGS.epsilon_1
# epsilon_1=np.inf
epsilon_2=FLAGS.epsilon_2

#计算敏感度
def construct_degs(graphs,adjs):
    degs = []
    for i in range(0, FLAGS.time_steps):
        G = graphs[i]
        deg = np.zeros((len(G.nodes()),))
        for nodeid in G.nodes():
            for jj in (G.neighbors(nodeid)):
                if FLAGS.sense=='sen':
                    deg[nodeid] = deg[nodeid]+adjs[i][nodeid,jj]
                else:
                    deg[nodeid] = deg[nodeid] + 1
        degs.append(deg)
    min_t = 0
    if FLAGS.window > 0:
        min_t = max(FLAGS.time_steps - FLAGS.window - 1, 0)
    return degs[min_t:]

#DP机制选择
def DP(mechanism,graph,adj,ep1,ep2,deg):
    if mechanism=='deg':
        print('epsilon_1:{},epsilon_2:{}'.format(ep1,ep2))
        graphs_n, adjs_n = load_noise_graphs_deg(graph, adj, ep1,ep2, deg)
    elif mechanism=='lp':
        print('epsilon_1:{}'.format(ep1))
        graphs_n, adjs_n = load_noise_graphs_lp(graph, adj, ep1)
    elif mechanism == 'gs':
        print('epsilon_1:{}'.format(ep1))
        graphs_n, adjs_n = load_noise_graphs_gs(graph, adj, ep1)
    elif mechanism =='geometric':
        graphs_n, adjs_n = geometric(graph,adj, ep1)
    elif mechanism =='exponential':
        graphs_n, adjs_n = exponential(graph,adj, ep1)
    return graphs_n, adjs_n

#本地差分隐私扰动
print('DP machine:{}'.format(FLAGS.machine))
if FLAGS.machine=='rr':
    graphs, adjs = load_graphs_Unweighted(FLAGS.dataset)
    graphs_n, adjs_n=load_noise_graphs_rr(graphs, adjs, epsilon_1)
    print('epsilon_1:{}'.format(FLAGS.epsilon_1))
else:
    graphs, adjs = load_graphs_npz(FLAGS.dataset)
    deg = construct_degs(graphs, adjs)
    # adj_norm = list(map(lambda adj: normalize_graph_gcn(adj), adjs))
    adj_norm = list(map(lambda adj: aug_random_walk(adj), adjs))
    graphs_n, adjs_n = DP(FLAGS.machine, graphs, adj_norm, epsilon_1, epsilon_2, deg)
    # graphs_n, adjs_n = load_noise(graphs, adj_norm, epsilon_1)

#构建训练验证测试集（3：1：1）
train_edges, train_edges_false,val_edges, val_edges_false, test_edges,test_edges_false=get_evaluation_data(adjs_n, num_time_steps)

#读取特征（默认自编码）
if FLAGS.featureless:
    print('Featureless,Using self coding features.')
    feats = [scipy.sparse.identity(adjs_n[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs_n if
             x.shape[0] <= adjs_n[num_time_steps - 1].shape[0]]  # 选择需要的时间点，创建one-hot特征
else:
    print('Read feature.')
    feats = load_feats(FLAGS.dataset)
num_features = feats[0].shape[1]
assert num_time_steps < len(adjs_n) + 1

adj_train = []
feats_train = []
num_features_nonzero = []
loaded_pairs = False

#上下文节点
context_pairs_train = get_context_pairs(graphs_n, num_time_steps)

#重构倒数第二张图
new_G = nx.MultiGraph()
new_G.add_nodes_from(graphs_n[num_time_steps - 1].nodes(data=True))
for e in graphs_n[num_time_steps - 2].edges():
    new_G.add_edge(e[0], e[1])
graphs_n[num_time_steps - 2] = new_G
adjs_n[num_time_steps - 2] = nx.adjacency_matrix(new_G)
feats[num_time_steps - 2]=feats[num_time_steps - 1]

adj_train_n = list(map(lambda adj: normalize_graph_gcn_2(adj), adjs_n))

num_features = feats[0].shape[1]
feats_train = list(map(lambda feat: preprocess_features(feat)[1], feats))
num_features_nonzero = [x[1].shape[0] for x in feats_train]

#初始化
def construct_placeholders(num_time_steps):
    min_t = 0
    if FLAGS.window > 0:
        min_t = max(num_time_steps - FLAGS.window - 1, 0)
    placeholders = {
        'node_1': [tf.placeholder(tf.int32, shape=(None,), name="node_1") for _ in range(min_t, num_time_steps)],
        'node_2': [tf.placeholder(tf.int32, shape=(None,), name="node_2") for _ in range(min_t, num_time_steps)],
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),  # [None,1]
        'features': [tf.sparse_placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                     range(min_t, num_time_steps)],
        'adjs': [tf.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for i in
                 range(min_t, num_time_steps)],
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='spatial_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='temporal_drop')
    }
    return placeholders

#tensorfflow加载模型
print("Initializing session")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
placeholders = construct_placeholders(num_time_steps)
minibatchIterator = NodeMinibatchIterator(graphs_n, feats_train, adj_train_n,
                                          placeholders, num_time_steps, batch_size=FLAGS.batch_size,
                                          context_pairs=context_pairs_train)
print("# training batches per epoch", minibatchIterator.num_training_batches())
model = DG(placeholders, num_features, num_features_nonzero, minibatchIterator.degs)
sess.run(tf.global_variables_initializer())
epochs_test_result = defaultdict(lambda: [])
epochs_val_result = defaultdict(lambda: [])
epochs_embeds = []
epochs_attn_wts_all = []
saver = tf.train.Saver()

#开始训练
for epoch in range(FLAGS.epochs):
    minibatchIterator.shuffle()
    epoch_loss = 0.0
    it = 0
    print('Epoch: %04d' % (epoch + 1))
    epoch_time = 0.0
    while not minibatchIterator.end():
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        feed_dict.update({placeholders['spatial_drop']: FLAGS.spatial_drop})
        feed_dict.update({placeholders['temporal_drop']: FLAGS.temporal_drop})
        t = time.time()
        _, train_cost, graph_cost, reg_cost = sess.run([model.opt_op, model.loss, model.graph_loss, model.reg_loss],
                                                       feed_dict=feed_dict)
        epoch_time += time.time() - t
        logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, train_cost))
        logging.info("Mini batch Iter: {} graph_loss= {:.5f}".format(it, graph_cost))
        logging.info("Mini batch Iter: {} reg_loss= {:.5f}".format(it, reg_cost))
        logging.info("Time for Mini batch : {}".format(time.time() - t))
        epoch_loss += train_cost
        it += 1
    print("Time for epoch ", epoch_time)
    logging.info("Time for epoch : {}".format(epoch_time))
    #预测任务
    if (epoch + 1) % FLAGS.test_freq == 0:
        minibatchIterator.test_reset()
        emb = []
        feed_dict.update({placeholders['spatial_drop']: 0.0})
        feed_dict.update({placeholders['temporal_drop']: 0.0})
        emb = sess.run(model.final_output_embeddings, feed_dict=feed_dict)[:,
              model.final_output_embeddings.get_shape()[1] - 2, :]
        emb = np.array(emb)
        #预测任务
        val_results,_,test_results, _ = evaluate_classifier(train_edges,
                                              train_edges_false,val_edges,
                                              val_edges_false, test_edges,
                                              test_edges_false, emb, emb)
        epoch_auc_val = val_results["LRC"][0]
        epoch_auc_test = test_results["LRC"][0]
        print("Epoch {}, Val AUC {}".format(epoch, epoch_auc_val))
        print("Epoch {}, Test AUC {}".format(epoch, epoch_auc_test))
        logging.info("Val results at epoch {}: Measure ({}) AUC: {}".format(epoch, "LRC", epoch_auc_val))
        logging.info("Test results at epoch {}: Measure ({}) AUC: {}".format(epoch, "LRC", epoch_auc_test))
        epochs_test_result["LRC"].append(epoch_auc_test)
        epochs_val_result["LRC"].append(epoch_auc_val)
        epochs_embeds.append(emb)
    epoch_loss /= it
    print("Mean Loss at epoch {} : {}".format(epoch, epoch_loss))

#找到最好的epoch再做一次测试
best_epoch = epochs_val_result["LRC"].index(max(epochs_val_result["LRC"]))
print("Best epoch ", best_epoch)
logging.info("Best epoch {}".format(best_epoch))
val_results,_,test_results, _ = evaluate_classifier(train_edges,
                                              train_edges_false,val_edges,
                                              val_edges_false, test_edges,
                                              test_edges_false, epochs_embeds[best_epoch],
                                                      epochs_embeds[best_epoch])
print("Best epoch test results {}\n".format(test_results))

logging.info("Best epoch val results {}\n".format(val_results))
logging.info("Best epoch test results {}\n".format(test_results))
write_to_csv(val_results, output_file, FLAGS.model, FLAGS.dataset, mod='val')
write_to_csv(test_results, output_file, FLAGS.model, FLAGS.dataset,mod='test')
saver.save(sess, MODEL_DIR+"/model.ckpt")
emb = epochs_embeds[best_epoch]
np.savez(SAVE_DIR + '/{}_DATA_{}.npz'.format(FLAGS.model, FLAGS.dataset), data=emb)
