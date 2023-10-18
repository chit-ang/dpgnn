import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags_dict=[]
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()

keys_list = [keys for keys in flags_dict]
print(keys_list)
keys = 'log_dir'
FLAGS.__delattr__(keys)
del_all_flags(tf.app.flags.FLAGS)

#隐私保护设置
flags.DEFINE_string('machine', 'deg', 'Differential privacy noise mechanism:{deg,lp,gs,rr}.')
flags.DEFINE_string('sense', 'sen', 'Sensitivity setting:{sen,deg}.')
flags.DEFINE_float('epsilon_1', 3, 'Noise related budget') #float('inf')
flags.DEFINE_float('epsilon_2', 4, 'Bernoulli distribution related budget')

#训练设置
flags.DEFINE_string('base_model', 'DG', 'Base model string')
flags.DEFINE_string('model', 'default', 'Model string.')
flags.DEFINE_string('dataset', 'ML-10M', 'Dataset string.')
flags.DEFINE_integer('time_steps', 13, 'time steps to train (+1)') # Predict at next time step.
flags.DEFINE_integer('GPU_ID', 0, 'GPU_ID')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 512, 'Batch size (# nodes)')
flags.DEFINE_boolean('featureless', True, 'Use 1-hot instead of features')
flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for self-attention model.')
flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
flags.DEFINE_integer('seed', 7, 'Random seed')

# 正负样本设置
flags.DEFINE_integer('neg_sample_size', 10, 'number of negative samples')
flags.DEFINE_integer('walk_len', 20, 'Walk len')
flags.DEFINE_float('neg_weight', 1.0, 'Wt. for negative samples')

#网络层设置
flags.DEFINE_float('spatial_drop', 0.1, 'attn Dropout (1 - keep probability).')
flags.DEFINE_float('temporal_drop', 0.5, 'ffd Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_boolean('use_residual', False, 'Residual connections')
flags.DEFINE_string('structural_head_config', '16,8,8', 'Encoder layer config: # attention heads in each GAT layer')
flags.DEFINE_string('structural_layer_config', '128', 'Encoder layer config: # units in each GAT layer')
flags.DEFINE_string('temporal_head_config', '16', 'Encoder layer config: # attention heads in each GAT layer')
flags.DEFINE_string('temporal_layer_config', '128', 'Encoder layer config: # units in each GAT layer')
flags.DEFINE_boolean('position_ffn', True, 'Use position wise feedforward')

#结果输出设置
flags.DEFINE_string('save_dir', "output", 'Save dir defaults to output/ within the base directory')
flags.DEFINE_string('log_dir', "log", 'Log dir defaults to log/ within the base directory')
flags.DEFINE_string('csv_dir', "csv", 'CSV dir defaults to csv/ within the base directory')
flags.DEFINE_string('model_dir', "model", 'Model dir defaults to model/ within the base directory')
flags.DEFINE_integer('window', -1, 'Window for temporal attention')
flags.DEFINE_integer('test_freq', 1, 'Testing frequency')
flags.DEFINE_integer('val_freq', 1, 'Validation frequency')

