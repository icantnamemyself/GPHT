from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Pretrain
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'pretrain': Dataset_Pretrain,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'traffic': Dataset_Custom,
    'electricity': Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'weather': Dataset_Custom,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag, data=None):
    Data = data_dict[args.data] if not data else data_dict[data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'ar_test':
        shuffle_flag = False
        drop_last = False
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
    size = [args.ar_seq_len, args.label_len, args.ar_pred_len] if flag == 'ar_test' else [args.seq_len,
                                                                                          args.label_len,
                                                                                          args.pred_len]
    Data = data_dict[args.transfer_data if flag == 'ar_test' else args.data]
    batch_size = args.pretrain_batch_size if args.data == 'pretrain' and flag != 'ar_test' else args.batch_size
    data_set = Data(
        root_path=args.transfer_root_path if flag == 'ar_test' else args.root_path,
        data_path=args.transfer_data_path if flag == 'ar_test' else args.data_path,
        flag=flag,
        size=size,
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        percent=args.percent
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
