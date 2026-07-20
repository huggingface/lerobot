from torch.nn.parallel import DistributedDataParallel as DDP

from . import model_utils as mutil

def calc_chan(branch_in_list, grayscale=False):
    indv_ch = 1 if grayscale else 3
    ch_dict = {'polar': indv_ch, 
               'mask': 1, 
               'normal': 3,
               'stokes': 6, 
    }
    return sum(ch_dict[k] for k in branch_in_list)

def init_input_chans(model_in_list, grayscale=False):
    if len(model_in_list) == 1:
        in_chan = calc_chan(model_in_list[0], grayscale)
    else:
        in_chan = [calc_chan(in_list, grayscale) for in_list in model_in_list]
    return in_chan

def build_normal_model(args, stage=None):
    print(f'Creating Normal Model [{args.normal_model}]') 
    in_chans = init_input_chans(args.normal_branch_inputs, args.grayscale)
    models = __import__(f'models.{args.normal_model}')
    model_file = getattr(models, args.normal_model)
    model = getattr(model_file, 'generate_model')(in_chans)
    
    if args.multi_gpu:
        model = DDP(model.to(args.device), device_ids=[args.local_rank], broadcast_buffers=False)
    else:
        model = model.to(args.device)

    if args.normal_pretrain and stage is None:
        args.log.wprint(f'#### Pretrained Normal Model: {args.normal_pretrain} ####', True)
        mutil.load_checkpoint(args.normal_pretrain, model)
        
    if args.normal_resume and stage is None:
        args.log.wprint(f'#### Resume Normal Model: {args.normal_resume} ####', True)
        mutil.load_checkpoint(args.normal_resume, model)
        args.resume = args.normal_resume

    args.log.write(f'{str(model)}', True)
    args.log.wprint(f'=> Normal #Parameters : {mutil.get_model_params(model)}', True)
    return model
