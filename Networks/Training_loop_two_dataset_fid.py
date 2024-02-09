import copy
import json
import os
import pickle
import time

import Dnnlib
import numpy as np
import cv2
import psutil
import torch

from Metrics import metric_main
from Dataset_utils.Samplers import Inf_Sampler
from Torch_utils import misc, training_stats
from Torch_utils.ops import conv2d_gradfix, grid_sample_gradfix


import matplotlib.pyplot as plt

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed) # 与其他生成器隔离的伪随机数生成器
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32) # 生成图像网格的大小，最小包含7×4张图，最大包含32×32张图
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices) # 随机打乱索引
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)] # 从被打乱的索引中抽取所需图像数

    # Load data.
    images = [training_set.get_paired_ImgNmap(i)[0].numpy() for i in grid_indices] # 将所需的数据打包
    nmaps = [training_set.get_paired_ImgNmap(i)[1].numpy() for i in grid_indices]
    return (gw, gh), images, nmaps


#----------------------------------------------------------------------------

def save_image_grid(nmaps, imgs, fname, grid_size):

    pairs = []
    for (img, nmap) in zip(imgs, nmaps):
        img = np.concatenate([(nmap+1)*127.5, (img+1)*127.5], axis=1)
        img = img.transpose(1,2,0).astype(int).clip(0, 255)
        pairs.append(img)


    gw, gh = grid_size
    count = 0
    rows = []; columns = []
    for idx in range(gw*gh):
        count += 1
        rows.append(pairs[idx])
        if count >= gw:
            columns.append(np.hstack(rows))
            rows = []; count=0
        

    cv2.imwrite(fname, np.vstack(columns)[:,:,::-1])



#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.

    training_set_kwargs     = {},       # Options for training set.
    other_training_set_kwargs={},
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.

    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.

    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.

    loss_kwargs             = {},       # Options for loss function.

    metrics                 = [],       # Metrics to evaluate during training.

    random_seed             = 0,        # Global random seed.
    rank                    = 0,        # Rank of the current process in [0, num_gpus].
    num_gpus                = 1,        # Number of GPUs participating in the training.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.


    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.

    resume_pkl              = None,     # Network pickle to resume training from.

    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?

):

    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank) # 每张GPU设置不同随机种子
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    conv2d_gradfix.enabled = True                       # Improves training speed.as s
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.


    # Load training set.
    if rank == 0:
        print('Loading training set...')

    training_set = Dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = Inf_Sampler(data_source=training_set, process_id=rank, num_processes=num_gpus, need_shuffle=True, random_seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print()


    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    G = Dnnlib.util.construct_class_by_name(**G_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = Dnnlib.util.construct_class_by_name(**D_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    networks_kwargs = dict(G=dict(**G_kwargs), D=dict(**D_kwargs))


    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming model from {resume_pkl}')
        with open(resume_pkl, 'rb') as f:
            params_data = pickle.load(f)
            f.close()
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            module.load_state_dict(params_data[name])


    # Print network summary tables.
    if rank == 0:
        z = torch.empty([1, G.z_dim], device=device)
        nmap = torch.empty([1, G.img_channels, G.img_resolution, G.img_resolution], device=device)
        img = misc.print_module_summary(G, [z, nmap])
        misc.print_module_summary(D, [img])
        del z, nmap, img    


    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (ada_target is not None):
        augment_pipe = Dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(0))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')
            # ada_stats = training_stats.Collector(regex='Loss/signs/real_main') # fixme: D_branch_v1 结构时


    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module


    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = Dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss
    phases = [] # G_opt_kwargs：G优化器配置； G_reg_interval: lazy regularization 的间隔步长， D同。
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        mb_ratio = reg_interval / (reg_interval + 1)
        opt_kwargs = Dnnlib.EasyDict(opt_kwargs)
        opt_kwargs.lr = opt_kwargs.lr * mb_ratio
        opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
        opt = Dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer 初始化优化器
        phases += [Dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
        phases += [Dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    if rank == 0:
        phase_infos = [{phase.name: phase.interval} for phase in phases]
        print('Phases\'s interval infos: ', phase_infos)
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_nmaps = None # CHW 
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, grid_nmaps = setup_snapshot_image_grid(training_set=training_set)
        grid_z = torch.randn([grid_size[0] * grid_size[1], G.z_dim], device=device).split(1)
        save_image_grid(grid_nmaps, images, os.path.join(run_dir, 'reals.png'), grid_size=grid_size)

        images = []
        for idx in range(grid_size[0] * grid_size[1]):
            z = grid_z[idx]
            nmap = torch.from_numpy(grid_nmaps[idx]).to(device=device).unsqueeze(0)
            images.append(G_ema(z, nmap).squeeze(0).detach().cpu().numpy())
        save_image_grid(grid_nmaps, images, os.path.join(run_dir, 'fakes_init.png'), grid_size=grid_size)


    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')


    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    

    while True:
        # Fetch training data.
        phase_real_img, phase_normal_map = next(training_set_iterator) # batch_size * imgs
        phase_real_img = (phase_real_img.to(device).to(torch.float32)).split(batch_gpu) # 将数据归分批
        phase_normal_map = (phase_normal_map.to(device).to(torch.float32)).split(batch_gpu) 

        all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
        all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z in zip(phases, all_gen_z):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, gen_z, normal_map) in enumerate(zip(phase_real_img, phase_gen_z, phase_normal_map)):
                # sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phasename=phase.name, real_img=real_img, nmap=normal_map, gen_z=gen_z, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            for param in phase.module.parameters():
                if param.grad is not None:
                    misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))



        # Update state.
        cur_nimg += batch_size
        batch_idx += 1


        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            # adjust = np.sign(ada_stats['Loss/signs/real_main'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000) # fixme: D_branch_v1 结构时
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))


        # Perform maintenance tasks once per tick. 训练完指定大小的图片就跳出训练
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        

        # Update G_ema.
        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
            p_ema.copy_(p)
        for b_ema, b in zip(G_ema.buffers(), G.buffers()):
            b_ema.copy_(b)


        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {Dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))


        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = []
            for idx in range(grid_size[0] * grid_size[1]):
                z = grid_z[idx]
                nmap = torch.from_numpy(grid_nmaps[idx]).to(device=device).unsqueeze(0)
                images.append(G_ema(z, nmap).squeeze(0).detach().cpu().numpy())
            save_image_grid(grid_nmaps, images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), grid_size=grid_size)



        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        snapshot_params = None # 记录参数
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict()
            snapshot_params = dict(net_kwargs=networks_kwargs, training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_params[name] = module.state_dict() # 保存各个模块的参数与整体的网络设置
                snapshot_data[name] = module
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_params, f)
                    

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                  dataset_kwargs=training_set_kwargs, other_dataset_kwargs=other_training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory


        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()


        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()


        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
