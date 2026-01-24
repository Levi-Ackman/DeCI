import argparse
import torch
from exp.exp_classification_CV import Exp_Main
import random
import numpy as np
import os
from tqdm import tqdm
import psutil
def use_cpus(gpus: list, cpus_per_gpu: int):
        cpus = []
        for gpu in gpus:
            cpus.extend(list(range(gpu* cpus_per_gpu, (gpu+1)* cpus_per_gpu)))
        p = psutil.Process()
        p.cpu_affinity(cpus)
        print("A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(len(cpus)))
        
def seed_set(seed=2024):
    random.seed(seed)   
    np.random.seed(seed)   
    torch.manual_seed(seed)   

def train(ii,args):
    setting = '{}_data_type_{}_protocol_{}_kfold_{}_model_{}_bs_{}_lr_{}_dp_{}_dm_{}_seq_{}'.format(
                    args.data,
                    args.data_type,
                    args.protocol,
                    args.kfold,
                    args.model,
                    args.batch_size,
                    args.learning_rate,
                    args.dropout,
                    args.d_model,
                    args.seq_len,
                    )

    exp = Exp(args)  # set experiments
    print(f'Start K-Fold Cross Validation Training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    avg_metric=exp.kf_train(setting) if args.Method == 'DL'  else exp.kf_ML(setting)
    print(f'End Training : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    
    return avg_metric
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General Time Series Backbone for FMRI Classification')

    # basic config
    parser.add_argument('--Method', type=str, default='DL',
                        help='Using Mechine Learning (ML) or not, if True, then use ML model (choose form SVM/svm, RF/rf, etc.), else use deep learning model (DL)')
    parser.add_argument('--model', type=str, default='iTransformer',
                        help='Model, choose from [Linear, iTransformer]')
    # data loader
    parser.add_argument('--data', type=str, default='PPMI', help='Data used, choose from [PPMI,  Mātai, Neurocon, Taowu, Abide]')
    parser.add_argument('--data_path', type=str, default='/data/gqyu/FMRI/dataset/PPMI', help='path to data file')
    parser.add_argument('--data_type', type=str, default='TS', 
                        help='Data type, choose from [TS: raw time series, FC: functional connectivity]')
    parser.add_argument('--protocol', type=str, default="schaefer100", 
                        help='Protocol of BrainNet, choose from [schaefer100, AAL116, harvard48, ward100, kmeans100]. \
                        This also decided the number of ROIs (or Channels in multivariate time series)')
    parser.add_argument('--kfold', type=int, default=5, help='number of fold for K-Fold Cross Validation')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--del_weight', type=bool, default=True, help='Delete model weight or not, if True, then delete the model weight after testing to save space')

    # classification task
    parser.add_argument('--seq_len', type=int, default=210, 
                        help='input sequence length of FMRI data, choose from [PPMI: 210  Mātai: 200  Neurocon: 137  Taowu: 239  Abide: 120]. \
                        Notabley, the input sequence length is different when using FC, where seq_len=channel. ')
    parser.add_argument('--channel', type=int, default=100, 
                        help='Number of ROI (Region of Interest) in FMRI data, which is equivalent to channel/variate in regular multivarite time seires. \
                        choose from [schaefer100: 100  AAL116: 116  harvard48: 48  ward100: 100  kmeans100: 100]')
    parser.add_argument('--classes', type=int, default=4, 
                        help='categories in data, choose from [PPMI: 4  Mātai: 2  Neurocon: 2  Taowu: 2  Abide: 2]')
    # model define
    parser.add_argument('--d_model', type=int, default=128, help='model hidden dimension')
    parser.add_argument('--layer', type=int, default=2, help='number of time series backbone layers')
    parser.add_argument('--n_head', type=int, default=8, help='num of of heads of the muilti-head attention')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--use_norm', type=int, default=0, help='use Reversible Normalization')
    
    # Autoformer/FEDformer
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--moving_avg', type=int, default=25, help='kernel size of move average kernel used to seasonal-trend decomposition')
    
    #TimesNet
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=2, help='for Inception')
    
    
    # TimeMixer
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp (when use this, top_k need reset)')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    
    # SegRNN
    parser.add_argument('--seg_len', type=int, default=24,
                        help='the length of segmen-wise iteration of SegRNN')
    
    # ModernTCN
    parser.add_argument('--small_kernel_merged', type=bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
    parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5], help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+',type=int, default=[64,64,64], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[64,64,64], help='dw dims in dw conv in each stage')
    
    #Medformer
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="whether to use single channel patching for Medformer",
        default=False,
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="12,24,48",
        help="a list of patch len used in Medformer",
    )
    parser.add_argument(
        "--no_inter_attn",
        action="store_true",
        help="whether to use inter-attention in encoder, using this argument means not using inter-attention",
        default=False,
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    

    # optimization
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='CE', 
                        help='loss function, MSE/mse for Binary Classification (Mātai, Neurocon, Taowu, Abide), CrossEntopy Loss (CE/ce) for multiple classes (PPMI-4 fro example)')
    parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
    parser.add_argument('--print_process', type=int, default=0, help='print each epoch process')
    parser.add_argument('--print_data_info', type=int, default=0, help='print each epoch process')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0,1,2,3,4,5,6,7], help="List of GPU indices to use")
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')


    args = parser.parse_args()
    seed_set(args.seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # For server 1, set cpus_per_gpu to 12
    #For server 2, set cpus_per_gpu to 24
    use_cpus(gpus=args.gpu_idx, cpus_per_gpu=12)
    
    print('Args in experiment:')
    print(args)
    Exp = Exp_Main
    avg_metrics=[]
    means=[]
    stds=[]
    
    iteration=5
    for i in tqdm(range(iteration)):
        print(f'The {i+1}-th {args.kfold} CE training begin>>>>>>>>>>>>>>>>>>>\n')
        avg_metrics.append(train(1,args))
        print(f'The {i+1}-th {args.kfold} CE training end<<<<<<<<<<<<<<<<<<<<<\n')
    means=[np.mean([avg_metrics[i][j] for i in range(iteration)]) for j in range(5)]
    stds=[np.std([avg_metrics[i][j] for i in range(iteration)]) for j in range(5)]
    print(f'Mean accuracy: {means[0]:.4f}, precision: {means[1]:.4f},recall: {means[2]:.4f}, macro_f1: {means[3]:.4f}, roc_auc: {means[4]:.4f}')
    print(f'Std accuracy: {stds[0]:.4f}, precision: {stds[1]:.4f},recall: {stds[2]:.4f}, macro_f1: {stds[3]:.4f}, roc_auc: {stds[4]:.4f}')