import os
import mxnet as mx

# path
source_root = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
data_root = os.path.join(source_root, 'data')
model_root = os.path.join(source_root, 'data', 'model')
log_root = os.path.join(source_root, 'data', 'log')

if not os.path.exists(model_root):
    os.makedirs(model_root)
if not os.path.exists(log_root):
    os.makedirs(log_root)

# dictionary
bos_word = '<s>'
eos_word = '</s>'
unk_word = '<unk>' #OOV
special_words = {unk_word: 1, bos_word: 2, eos_word: 3}
source_vocab_path = os.path.join(data_root, 'src', 'src.vocab.pkl')
target_vocab_path = os.path.join(data_root, 'tgt', 'tgt.vocab.pkl')

# data set
train_source = os.path.join(data_root, 'src', 'src.txt')
train_target = os.path.join(data_root, 'tgt', 'tgt.txt')
train_max_samples = 100000
# Why the name is invalid for dev target?
dev_source = os.path.join(data_root, 'dev', 'dev_src.txt')
dev_target = os.path.join(data_root, 'invalid', 'invalid')
dev_output = os.path.join(data_root, 'dev', 'dev_tgt.txt')
dev_max_samples = 100000
test_source = os.path.join(data_root, 'test', 'test.txt')
test_gold = os.path.join(data_root, 'test', 'test.txt')

bleu_ref_number = 7

# model parameter
batch_size = 128
bucket_stride = 10
buckets = []
for i in range(10, 70, bucket_stride):
    for j in range(10, 70, bucket_stride):
        buckets.append((i, j))
num_hidden = 512  # hidden unit in LSTM cell
num_embed = 512  # embedding dimension
num_lstm_layer = 1  # number of lstm layer

# training parameter
num_epoch = 60
learning_rate = 1
momentum = 0.1
dropout = 0.5
show_every_x_batch = 100
eval_per_x_batch = 400
eval_start_epoch = 4

# model save option
model_save_name = os.path.join(model_root, 'seq_to_seq')
model_save_freq = 3  # every x epoch
checkpoint_name = os.path.join(model_root, 'checkpoint_model')
checkpoint_freq_batch = 1000  # save checkpoint model every x batch

# train device  Can we have more gpu here?
train_device = [mx.context.gpu(0)]
# test device
test_device = mx.context.gpu(0)

# test parameter
model_to_load_prefix = os.path.join(model_root, 'seq_to_seq')
model_to_load_number = 1
use_beam_search = True
beam_size = 12
if not use_beam_search: beam_size = 1
test_output = os.path.join(data_root, 'test', 'test.out')
use_batch_greedy_search = False
greedy_batch_size = 32
max_decode_len = 15

# resume training
use_resuming = True
resume_model_prefix = os.path.join(model_root, "checkpoint_model")
resume_model_number = 0


def get_config_str():
    res = ''
    res += 'Config:\n'
    import collections
    hehe = collections.OrderedDict(sorted(globals().items(), key=lambda x: x[0]))
    for k, v in hehe.items():
        if k.startswith('__'): continue
        if k.startswith('SEPARATOR'): continue
        if k.startswith('get'): continue
        if type(v) == (type(os)): continue
        if len(k) < 2: continue
        res += '{0}: {1}\n'.format(k, v)
    return res
