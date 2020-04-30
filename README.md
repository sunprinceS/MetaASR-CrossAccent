MetaASR CrossAccent
===

**這個 repo 請先不要外流**

## Prerequisites
### global
* torch==1.4.1
* numpy==1.18.3
* sentencepiece==0.1.85 (for subword units)

### local
* comet-ml==3.1.6, and you should register using ntu email
* editdistance==0.5.3
* tqdm-logger==0.3.0 [repo](https://github.com/leomao/tqdm-logger)
* torchexp==0.1.0 [repo](https://github.com/leomao/torchexp)
* torch_optimizer==0.0.1a11 [repo](https://github.com/jettify/pytorch-optimizer)

### misc
* Download corpus -> ask me
* modify the `COMET_PROJECT_NAME`, `COMET_WORKSPACE` in `src/marcos.py` to your setting

## Note
* One exp: (Pretraining) -> training (`train.py`) -> testing/decoding (`train.py --test`) -> `score.sh`
* Each trainer will instantialized with different interface to decide its training/pretraining behavior

## Structure

### root 
* `train.py`: 
    * mono-accent training (include training from scratch or fine-tune on pretrained model)
    * testing (aka decoding): add `--test` flag
* `pretrain.py`: multi-accent pretraining (thru meta/multi-task)
* excute `run_foo.sh` (it will call `foo_full_exp.sh` to conduct one complete experiment) to conduct large-scale experiments on battleship
* `score.sh` will call `translate.py` with proper env. variables to excute sctk to evaluate the error rate
* `data/`: soft link to data
* `config/`: config yaml
* `testing-logs/`: Can also modify the name in `src/marcos.py`
    * `pretrain`: store info when excuting `pretrain.py`
    * `evaluation`: store info when excuting `train.py`
* `tensorboard`: as title (but NOTE that we use [comet.ml](https://www.comet.ml/site/) to track the experiment, most of the time we don't need this)

### src
* main files
    * `foo_trainer.py`: define how to run one batch and some specific stuff for this model, will include `asr_model` inside
    * `tester.py`: define how to decode (all model use same tester)
    * `train_interface.py` and `mono_interface.py`: used in mono-accent training and fine-tuning
    * `pretrain_interface.py` and `fo_meta_interface.py`, `multi_interface.py`...: used in pretraining
* `modules/`: define some components in model (but a little bit deprecated now 😅)
* `io/`:
    * `data_loader`: used in mono-accent
    * `data_container` used in multi-accent
* `model/`:
    * transformer (NOTE: `transformer_pytorch`) is what we use
    * blstm (vgg-blstm fully-connected ctc model used in [MetaASR](https://arxiv.org/abs/1910.12094))
* `monitor`:
    * `dashboard.py` (comet.ml), `tb_dashboard` (tensorboard)
    * `logger.py`: tqdm_logger logging
    * `metric.py`: how to calculate error rate

