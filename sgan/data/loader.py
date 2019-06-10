from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, TrajectoryDataset_train, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,#cut into batches
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)#collate_fn用来处理不同情况下的输入dataset的封装，collate_fn (callable, optional) – merges a list of samples to form a mini-batch.！！！！！！
    
    return dset, loader

def data_loader_train(args, path):
    dset = TrajectoryDataset_train(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,#cut into batches
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)#collate_fn用来处理不同情况下的输入dataset的封装，collate_fn (callable, optional) – merges a list of samples to form a mini-batch.！！！！！！
    
    return dset, loader
#Dataset只负责数据的抽象，一次调用getitem只返回一个样本。前面提到过，在训练神经网络时，最好是对一个batch的数据进行操作，同时还需要对数据进行shuffle和并行加速等。对此，PyTorch提供了DataLoader帮助我们实现这些功能。
