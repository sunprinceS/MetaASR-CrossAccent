import torch 
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import src.monitor.logger as logger
import random

# from numpy.lib.format import open_memmap

BUCKET_SIZE=1
ILEN_MIN = 2
ILEN_MAX = 10000

# def _seed_worker(worker_idx):
    # seed = torch.initial_seed() & ((1 << 63) - 1)
    # random.seed(seed)
    # np.random.seed((seed >> 32, seed % (1 << 32)))

## Customized function
def collate_fn(batch):
    """
    batch list: samples
    """
    batch.sort(key=lambda d: d['ilen'], reverse=True)
    

    xs_pad = pad_sequence([d['feat'] for d in batch], batch_first=True)
    ilens = torch.stack([d['ilen'] for d in batch])
    ys = [d['label'] for d in batch]
    olens = torch.stack([d['olen'] for d in batch])

    return xs_pad, ilens, ys, olens

class BucketSampler(Sampler):
    def __init__(self, ilens, min_ilen, max_ilen, half_batch_ilen, \
                 batch_size, bucket_size, bucket_reverse, drop_last):

        self.ilens = ilens
        self.min_ilen = min_ilen
        self.max_ilen = max_ilen
        self.half_batch_ilen = half_batch_ilen if half_batch_ilen else ILEN_MAX
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.bucket_reverse =  bucket_reverse# if True: long -> short
        self._create_buckets()
        # logger.log(f"Bucket size distribution: {[len(bucket[1]) for bucket in self.buckets]}")

    def __iter__(self):
        for bin_idx, bucket in self.buckets:
            batch_size = self._get_batch_size(bin_idx)

            np.random.shuffle(bucket)
            batch = []
            for idx in bucket:
                batch.append(idx)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
        num_batchs = 0
        for bin_idx, bucket in self.buckets:
            batch_size = self._get_batch_size(bin_idx)

            if self.drop_last:
                num_batchs += len(bucket) // batch_size
            else:
                num_batchs += (len(bucket) + batch_size - 1) // batch_size

        return num_batchs

    def _get_batch_size(self,bin_idx):
        if self.bucket_reverse:
            batch_size = max(1, self.batch_size // 2) if bin_idx < self.half_batch_size_bucket_idx else self.batch_size
        else:
            batch_size = max(1, self.batch_size // 2) if bin_idx > self.half_batch_size_bucket_idx else self.batch_size

        return batch_size

    def _create_buckets(self):
        lb = min(ILEN_MIN,self.bucket_size) if not self.min_ilen else self.min_ilen
        ub = max(ILEN_MAX,self.ilens.max()) if not self.max_ilen else self.max_ilen
        if self.bucket_reverse:
            bins = np.arange(ub, lb, -self.bucket_size) # long -> short
        else:
            bins = np.arange(lb, ub, self.bucket_size) # short -> long

        bucket_idx = np.digitize(self.ilens, bins, right=True)
        self.half_batch_size_bucket_idx = np.digitize(self.half_batch_ilen, bins, right=True)
        self.buckets = []
        for bin_idx in range(1, len(bins) - 1):
            bucket = np.where(bucket_idx == bin_idx)[0]
            if len(bucket) > 0:
                self.buckets.append((bin_idx, bucket))
        # if self.bucket_reverse:
            # for bin_idx in range(1, len(bins)-1):
                # bucket = np.where(bucket_idx == bin_idx)[0]
                # if len(bucket) > 0:
                    # self.buckets.append((bin_idx, bucket))
        # else:
            # for bin_idx in range(1,len(bins)-1):
                # bucket = np.where(bucket_idx == bin_idx)[0]
                # if len(bucket) > 0:
                    # self.buckets.append((bin_idx, bucket))

        random.shuffle(self.buckets)


#TODO: In addition to npy and memmap choice, still need KaldiDataset


class CommonVoiceDataset(Dataset):

    def __init__(self, data_dir, is_memmap):
        """
        data_dir: str
        is_memmap: bool
        """
        if is_memmap:
            feat_path = data_dir.joinpath('feat').with_suffix('.dat')
            logger.log(f"Loading {feat_path} from memmap...",prefix='info')
            self.feat = np.load(feat_path, mmap_mode='r')
        else:
            feat_path = data_dir.joinpath('feat').with_suffix('.npy')
            logger.warning(f"Loading whole data ({feat_path}) into RAM")
            self.feat = np.load(feat_path)
        
        self.ilens = np.load(data_dir.joinpath('ilens.npy'))
        self.iptr = np.zeros(len(self.ilens)+1, dtype=int)
        self.ilens.cumsum(out=self.iptr[1:])

        self.label = np.load(data_dir.joinpath('label.npy'))
        self.olens = np.load(data_dir.joinpath('olens.npy'))
        self.optr = np.zeros(len(self.olens) + 1, dtype=int)
        self.olens.cumsum(out=self.optr[1:])

        assert len(self.ilens) == len(self.olens), \
        "Number of samples should be the same in features and labels"

    def __len__(self):
        return len(self.ilens)

    def __getitem__(self,idx):
        return{
            'feat':torch.as_tensor(self.feat[self.iptr[idx]:self.iptr[idx+1],:]),
            'ilen':torch.as_tensor(self.ilens[idx]),
            'label':torch.as_tensor(self.label[self.optr[idx]:self.optr[idx+1]]),
            'olen':torch.as_tensor(self.olens[idx]),
        }


def get_loader(data_dir, batch_size, is_memmap, is_bucket, num_workers=0, 
               min_ilen=None, max_ilen=None, half_batch_ilen=None, 
               bucket_reverse=False, shuffle=True, read_file=False, 
               drop_last=False, pin_memory=True):

    assert not read_file, "Load from Kaldi ark haven't been implemented yet"
    dset = CommonVoiceDataset(data_dir, is_memmap)

    # if data is already loaded in memory
    if not is_memmap: 
        num_workers = 0

    logger.notice(f"Loading data from {data_dir} with {num_workers} threads")

    if is_bucket:
        my_sampler = BucketSampler(dset.ilens, 
                                   min_ilen = min_ilen, 
                                   max_ilen = max_ilen, 
                                   half_batch_ilen = half_batch_ilen, 
                                   batch_size=batch_size, 
                                   bucket_size=BUCKET_SIZE, 
                                   bucket_reverse=bucket_reverse, 
                                   drop_last = drop_last)

        loader = DataLoader(dset, batch_size=1, num_workers=num_workers,
                            collate_fn=collate_fn, batch_sampler=my_sampler,
                            drop_last=drop_last, pin_memory=pin_memory)
    else:
        loader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=collate_fn, shuffle=shuffle,
                            drop_last=drop_last, pin_memory=pin_memory)

    return loader

# class DataContainer:
    # def __init__(self, data_dirs, batch_size, dev_batch_size, is_memmap, 
                 # is_bucket, num_workers=0, min_ilen=None, max_ilen=None, 
                 # half_batch_ilen=None, bucket_reverse=False, shuffle=True, 
                 # read_file=False, drop_last=False, pin_memory=True):

        # self.data_dirs = data_dirs
        # self.num_datasets = len(self.data_dirs)
        # self.batch_size = batch_size
        # self.is_memmap = is_memmap
        # self.is_bucket = is_bucket
        # self.num_workers = num_workers
        # self.min_ilen = min_ilen
        # self.max_ilen = max_ilen
        # self.half_batch_ilen = half_batch_ilen
        # self.bucket_reverse=bucket_reverse
        # self.shuffle = shuffle
        # self.read_file = read_file
        # self.reload_cnt = 0

        # self.loader_iters = list()
        # self.dev_loaders = list()

        # for data_dir in self.data_dirs:
            # self.loader_iters.append(
                # iter(get_loader(
                    # data_dir.joinpath('train'),
                    # batch_size = self.batch_size,
                    # is_memmap = self.is_memmap,
                    # is_bucket = self.is_bucket,
                    # num_workers = self.num_workers,
                    # min_ilen = self.min_ilen,
                    # max_ilen = self.max_ilen,
                    # half_batch_ilen = self.half_batch_ilen,
                    # bucket_reverse = self.bucket_reverse,
                    # shuffle = self.shuffle,
                    # read_file = self.read_file
            # )))
            # self.dev_loaders.append(
                # get_loader(
                # data_dir.joinpath('dev'),
                # batch_size = dev_batch_size,
                # is_memmap = self.is_memmap,
                # is_bucket = False,
                # num_workers = self.num_workers,
                # shuffle =False,
            # ))

    # def get_item(self, lang_idx=None, num=1):
        # ret_ls = []
        # if lang_idx is None: # for MultiASR
            # lang_ids = np.random.randint(self.num_datasets, size=num)
        # else:
            # lang_ids = np.repeat(lang_idx,num)

        # for lang_id in lang_ids:
            # try:
                # ret = next(self.loader_iters[lang_id])
                # ret_ls.append((lang_id,ret))
            # except StopIteration:
                # self.loader_iters[lang_id] = iter(get_loader(
                # self.data_dirs[lang_id].joinpath('train'),
                # batch_size = self.batch_size,
                # is_memmap = self.is_memmap,
                # is_bucket = self.is_bucket,
                # num_workers = self.num_workers,
                # min_ilen = self.min_ilen,
                # max_ilen = self.max_ilen,
                # half_batch_ilen = self.half_batch_ilen,
                # bucket_reverse = self.bucket_reverse,
                # shuffle = self.shuffle,
                # read_file = self.read_file))

                # self.reload_cnt += 1
                # ret = next(self.loader_iters[lang_id])
                # ret_ls.append((lang_id,ret))

        # return ret_ls

# if __name__ == "__main__":
    # data_dir = 'mydata/eval'

    # loader = get_loader(data_dir, batch_size=128, is_memmap=True, num_workers=4)

    # for data in loader:
        # print(data['feats'][-1])
