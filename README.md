# MXNMT: MXNet based Neural Machine Translation

This is an implementation seq2seq with attention for neural machine translation with MXNet.


## How to use it?
1. Generate Data
    python3.5 ../nmt/dict_gen.py ../data/src/src.txt --dictionary ../data/src.vocab.pkl
    python3.5 ../nmt/dict_gen.py ../data/tgt/tgt.txt --dictionary ../data/tgt.vocab.pkl
2. Train
    python3.5 ../nmt/main.py train
3. Test
    python3.5 ../nmt/main.py test
4. Service
    python3.5 ../nmt/main.py service

## ToDo
    Tune the configuration in xconfig.py to get better result
    [Deprecation Warning] mxnet.model.FeedForward has been deprecated. Please use mxnet.mod.Module instead.
