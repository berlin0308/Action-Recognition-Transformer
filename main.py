from pathlib import Path
import json
import random
import os
from torch import nn
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MultiMarginLoss
from torch.optim import SGD, lr_scheduler, Adam
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data
from torch.backends import cudnn
import torchvision
import torchvision.transforms as transforms


from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset import get_training_data, get_validation_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
from validation import val_epoch
import inference

from fvcore.nn import FlopCountAnalysis, parameter_count_table

from videomae_model.optim_factory import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from videomae_model.engine_for_finetuning import train_one_epoch
from videomae_model.utils import cosine_scheduler, TensorboardLogger
from videomae_model.utils import NativeScalerWithGradNormCount as NativeScaler
import math


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()        


    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path

        if opt.inf_day_eval:
            opt.result_path = Path("result_day/inf_day_eval")
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
        elif opt.inf_test_data:
            opt.result_path = Path("result_test")
        else:
            # Add on 11/10
            opt.result_path = Path(str(opt.result_path) +str(f"/{opt.model}_V8_b{opt.batch_size}_lr{opt.learning_rate}"))
            try:
                os.mkdir(opt.result_path)
            except:
                print("result folder exist !")
        
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth) # original: opt.svm
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]


    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:  # opt的紀錄 存在result_path/opts.json
                json.dump(vars(opt), opt_file, default=json_serial) # call json_serial，把opt寫成json檔案
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)
            
    return opt

def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']
    
    # print(checkpoint['arch'])

    # model.load_state_dict(checkpoint['state_dict'])
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'],strict=False)  # "strict=False" revised 

    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)

    

def get_train_utils(opt, model_parameters):
    assert opt.train_crop in ['random', 'corner', 'center']  # 裁切圖片
    spatial_transform = []                                   # 先取得spatial的參數
    if opt.train_crop == 'random':
        spatial_transform.append(RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5, hue=0.05))
    
    spatial_transform.append(ToTensor())
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']       # 取得temporal參數
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = get_training_data(opt.video_path, opt.annotation_path,
                                   opt.dataset, opt.input_type, opt.file_type,
                                   spatial_transform, temporal_transform)

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True, # revised: 20231112 
                                               num_workers=opt.n_threads,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn,
                                               pin_memory=True) # revised: 20230604

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:     # 某種優化演算法
        dampening = 0
    else:
        dampening = opt.dampening   # 阻尼梯度下降(SGD)
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    assert opt.lr_scheduler in ['plateau', 'multistep']  # 查一下什麼意思
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)  # milestones:遞增的list 內部存放要更新lr的epoch

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)

    val_data, collate_fn = get_validation_data(opt.video_path,
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,
                                               spatial_transform,
                                               temporal_transform)
    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size //
                                                         opt.n_val_samples),
                                             shuffle=True, # revised: 20231112 
                                             num_workers=opt.n_threads,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn,
                                             pin_memory=True) # revised: 20230604

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None

    return val_loader, val_logger


def get_inference_utils(opt):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    inference_data, collate_fn = get_inference_data(
        opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
        opt.file_type, opt.inference_subset, spatial_transform,
        temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=True) # revised: 20230604

    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    num_train_data = np.array([457, 657, 373, 495, 477, 297, 409]) # Dataset V8
    weights = torch.tensor(num_train_data.sum() / num_train_data, dtype=torch.float)
    print("weights:",weights)
    
    # apply label smoothing
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1).to(opt.device)




    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')
        print(str(opt.device.type))
    if opt.distributed:
        print(111111111111111111111111111111111111111111111)
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)  # 產生要用的模型之後可能可以從這邊著手改
    if opt.batchnorm_sync:      # 能夠支援多卡訓練
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'  # 警告說多卡訓練只能在前面使用多線程模式下執行
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        print(22222222222222222222222222222222222222222)       # 如果有pretrain_path的話,就把model更改成pretrain的
        model = load_pretrained_model(model, opt.pretrain_path, opt.model,  # 要注意這邊新增一個time feature的option
                                      opt.n_finetune_classes, opt.time)
    if opt.resume_path is not None:   # 給inference使用的
        model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.distributed, opt.device)  # 轉換model做平行運算使用

    if opt.pretrain_path:       # 取得fine tuning的layer 通常是 layer4 或是 fcl
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    else:
        parameters = model.parameters()

    if opt.is_master_node:    # opt.is_master_node = not opt.distributed or opt.dist_rank == 0
        pass
        # print(model)


    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node: # TensorBoard: 視覺化模型訓練過程
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    """
    	VideoMAEv2-b
        32x224x224
        -------------------------
        backbone	        Vit-B
        pre-train dataset	K710
        optimizer	        AdamW #
        base lr             [--learning_rate]
        weight decay	    0.05 #
        optimizer momentum	0.9 / 0.999 #
        batch size	        8
        lr scheduler	    cosine annealing #
        warmup epoch	    5 -> X #
        epoch	            [--n_epoch]
        repeated aug	    2
        random aug	        (0, 0.5)
        mixup, cutmix	    None #
        drop path	        0.35
        spatial aug	        RandomResizeCrop #
        dropout	            0.5
        layer-wise lr decay	0.9
    
    """

    if opt.official_hyper_params:

        # update_freq = 1
        # num_tasks = 1 # world size (不確定)

        """
        learning rate
        """

        # min_lr = 1e-5
        # warmup_lr = 1e-5
        # warmup_epochs = 5
        # warmup_steps = 1
        # weight_decay = 0.05
        # weight_decay_end = None

        # total_batch_size = opt.batch_size * update_freq * num_tasks
        # num_training_steps_per_epoch = len(train_loader) // total_batch_size
        # lr = lr * total_batch_size / 256
        # #########scale the lr#############
        # min_lr = min_lr * total_batch_size / 256
        # warmup_lr = warmup_lr * total_batch_size / 256
        # #########scale the lr#############
        # print("LR = %.8f" % lr)
        # print("Batch size = %d" % total_batch_size)
        # print("Update frequent = %d" % update_freq)
        # print("Number of training examples = %d" % len(train_loader))
        # print("Number of training training per epoch = %d" %
        #     num_training_steps_per_epoch)


        # Optimizer: AdamW
        # layer_decay = 0.99
        # num_layers = 15 #model.get_num_layers()
        # assigner = LayerDecayValueAssigner(
        #     list(layer_decay**(num_layers + 1 - i)
        #         for i in range(num_layers + 2)))
       
        # if assigner is not None:
        #     print("Assigned values = %s" % str(assigner.values))

        # optimizer = create_optimizer(
        #     lr,
        #     model,
        #     skip_list=None,
        #     get_num_layer=assigner.get_layer_id
        #     if assigner is not None else None,
        #     get_layer_scale=assigner.get_scale
        #     if assigner is not None else None)

        # weight_decay = 0.001
        # eps = 1e-8
        # betas = (0.9, 0.999)
        # optimizer = torch.optim.AdamW(params=model.parameters(),lr=opt.learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # Scheduler: Cosine Annealing
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        # Scheduler: cosine decay with warmup
        t = 0 # warmup epoch
        T = 100 # epochs
        n_t = 0.5
        cosine_decay_with_warmup = lambda epoch: (0.9*epoch / t +0.1) if epoch < t else 0.1 if n_t *(1+math.cos(math.pi*(epoch-t)/(T-t)))<0.1 else n_t *(1+math.cos(math.pi*(epoch-t)/(T-t)))
        scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=cosine_decay_with_warmup)


        
        # Loss Function: Weighted Cross Entropy with label smoothing
        smoothing = 0.1
        criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing).to(opt.device)


        # loss_scaler = NativeScaler()
        # print("Use step level LR scheduler!")
        # lr_schedule_values = cosine_scheduler(
        #     lr,
        #     min_lr,
        #     opt.n_epochs,
        #     num_training_steps_per_epoch,
        #     warmup_epochs=warmup_epochs,
        #     warmup_steps=warmup_steps,
        #     start_warmup_value=warmup_lr
        # )
        # if weight_decay_end is None:
        #     weight_decay_end = weight_decay
        # wd_schedule_values = cosine_scheduler(weight_decay,
        #                                             weight_decay_end,
        #                                             opt.n_epochs,
        #                                             num_training_steps_per_epoch)
        # print("Max WD = %.7f, Min WD = %.7f" %
        #     (max(wd_schedule_values), min(wd_schedule_values)))


    for i in range(opt.begin_epoch, opt.n_epochs + 1):  # Start Training
        if not opt.no_train:

            if i ==1: # check functionalities
                # dummy_x = torch.rand(1, 3, 32, 224, 224)
                # model.eval()
                # logits = model(dummy_x)
                # # print(logits.shape)

                # flops = FlopCountAnalysis(model, dummy_x)
                # print(f'Model FLOPs: {flops.total()}')
                print(f'Model Parameters: {parameter_count_table(model)}')
                
                # inference.inf_test_data(opt, model)
                # inference.inf_day_eval(opt, model)

            # if opt.distributed:
            #     train_sampler.set_epoch(i)

            if not opt.official_hyper_params:
                current_lr = get_lr(optimizer)
                train_loss = train_epoch(i, train_loader, model, criterion, optimizer,
                                        opt.device, current_lr, train_logger,
                                        train_batch_logger, tb_writer
                                        , opt.distributed, opt.time)
            else:
                current_lr = get_lr(optimizer) # AdamW
                train_loss = train_epoch(i, train_loader, model, criterion, optimizer,
                                        opt.device, current_lr, train_logger,
                                        train_batch_logger, tb_writer
                                        , opt.distributed, opt.time)

                # log_dir = os.path.join(opt.result_path, "log")
                # os.makedirs(log_dir, exist_ok=True)
                # log_writer = TensorboardLogger(log_dir=log_dir)
                # if log_writer is not None:
                #     log_writer.set_step(i * num_training_steps_per_epoch * update_freq)
                #     train_stats = train_one_epoch(
                #         model,
                #         criterion,
                #         train_loader,
                #         optimizer,
                #         opt.device,
                #         i,
                #         loss_scaler,
                #         None,
                #         model_ema=None,
                #         mixup_fn=None,
                #         log_writer=log_writer,
                #         start_steps=i * num_training_steps_per_epoch,
                #         lr_schedule_values=lr_schedule_values,
                #         wd_schedule_values=wd_schedule_values,
                #         num_training_steps_per_epoch=num_training_steps_per_epoch,
                #         update_freq=update_freq,
                #     )


            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = Path(str(opt.result_path) + f'/{opt.model}_V8_b{opt.batch_size}_lr{opt.learning_rate}_ep{i}.pth')
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)
                
                inference.inf_test_data(opt, model)
                inference.inf_day_eval(opt, model)


        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer,
                                      opt.distributed)

        
        scheduler.step()

            
    if opt.ipcam:
        inference_loader, inference_class_names = get_inference_utils(opt)
        inference_result_path = opt.result_path / '{}.json'.format(
            opt.inference_subset)

        inference.ipcam_main(opt, model)
        
    if opt.inference: # inference time!!!
        inference_loader, inference_class_names = get_inference_utils(opt)
        inference_result_path = opt.result_path / '{}.json'.format(
            opt.inference_subset)

        inference.inference(inference_loader, model, inference_result_path,
                            inference_class_names, opt.inference_no_average,
                            opt.output_topk, opt.time)
    
    if opt.inf_day:
        inference.inf_day_main(opt, model)
    
    if opt.inf_test_data:
        inference.inf_test_data(opt, model)
    
    if opt.inf_day_eval:
        inference.inf_day_eval(opt, model, csv=True)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    opt = get_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3' # 原本是'0, 2'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:256'
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    
    """ 
    distributed training->分佈式訓練
    若opt.distributed設定為true, multiprocessing
    若否則single-process training
    """
    if opt.distributed: 
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)
