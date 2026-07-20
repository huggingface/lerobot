import torch
from torch.nn.parallel import DistributedDataParallel, DataParallel
import torch.nn.functional as F

def load_data(sample, ddev):
    data = {}
    for item in sample:
        if item == 'polar':
            data_loaded = sample[item].to(device=ddev, non_blocking=True)
            data_loaded = data_loaded.permute(0, 1, 4, 2, 3)
        elif item == 'normal_gt':
            normal_gt = sample[item].to(device=ddev, non_blocking=True)
            normal_gt = normal_gt * 2 - 1
            data_loaded = F.normalize(normal_gt)
        elif item == 'name' or item == 'roi':
            data_loaded = sample[item]
        else:
            data_loaded = sample[item].to(device=ddev, non_blocking=True)
        data[item] = data_loaded
    return data

def get_inputs(data, branch_input_names):
    data['resize_candidate_inputs'] = []
    model_inputs = []
    for names in branch_input_names:
        input_dict = {}
        for name in names:
            if 'stokes' in name:
                s0 = data['polar'].mean(dim=1)
                s1 = ((data['polar'][:, 0] - data['polar'][:, 2]) / 2.)
                s2 = ((data['polar'][:, 1] - data['polar'][:, 3]) / 2.)
                data['stokes'] = torch.stack([s0, s1, s2], 1)
                if name == 'stokes':
                    stokes = data['stokes'].mean(dim=2) # (B,3,H,W)
                    aolp = 0.5*torch.atan2(stokes[:,2:3],stokes[:,1:2]+1e-8)
                    aolp_embed = torch.cat([torch.sin(aolp*2), torch.cos(aolp*2)], 1)
                    dolp = torch.sqrt(stokes[:,2:3]**2+stokes[:,1:2]**2)/(stokes[:,:1]+1e-8)
                    input_dict[name] = torch.cat([stokes, aolp_embed, dolp], 1) # (B,6,H,W)
                else:
                    raise ValueError(f'Unknown stokes input "{name}"')
            elif name == 'unpolar':
                data['unpolar'] = data['polar'].mean(dim=1)
                input_dict[name] = data['unpolar']
                data['resize_candidate_inputs'].append('unpolar')
            elif name == 'polar_cat':
                num = data['polar'].shape[1]
                polars = [data['polar'][:,i] for i in range(num)]
                input_dict[name] = torch.cat(polars, 1)
                data['resize_candidate_inputs'].append('polar')
            elif name in ['polar', 'mask']:
                input_dict[name] = data[name]
                data['resize_candidate_inputs'].append(name)
            else:
                raise Exception(f'Unknown input name: {name}')
            if len(input_dict[name].shape) == 4:
                input_dict[name] = input_dict[name]*data['mask'].detach()
            elif len(input_dict[name].shape) == 5:
                input_dict[name] = input_dict[name]*data['mask'][:,None].detach()
            else:
                raise Exception(f'Data {name} has the shape of {input_dict[name].shape}')
        model_inputs.append(input_dict)
    if len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    return model_inputs

def update_data(data, pred, split=None):
    data['resize_candidate_outputs'] = ['mask']
    for name in pred:
        if name == 'normal':
            pred[name] = F.normalize(pred[name])
        data[f'{name}_pred'] = pred[name]

        if split == 'test':
            data[f'{name}_pred'] = data[f'{name}_pred']*data['mask'].detach()
        data['resize_candidate_outputs'].append(f'{name}_pred')

def postprocess_data(data, split='train'):
    if 'roi' in data:
        if split == 'train':
            resize_seq_names = data['resize_candidate_outputs']
        elif split == 'test':
            resize_seq_names = data['resize_candidate_outputs'] + data['resize_candidate_inputs']
        else:
            raise ValueError(f'Unknown split: {split}')
        resize_names = list(set(resize_seq_names))
        resize_names.sort(key=resize_seq_names.index)
        roi = data['roi']
        B = roi.shape[0]
        h0, w0 = roi[0,:2]
        dev = data['mask'].device
        for name in resize_names:
            zero_size = [i for i in data[name].shape[:-2]] + [h0, w0]
            out_tensor = torch.zeros(zero_size, device=dev)
            t_size_num = len(out_tensor.shape)
            for i in range(B):
                h0, w0, rs, re, cs, ce = roi[i]
                if t_size_num == 4:
                    tensor = data[name][i:i+1]
                elif t_size_num == 5:
                    tensor = data[name][i]
                else:
                    Exception()
                tensor = F.interpolate(tensor, size=(re-rs, ce-cs), mode='bicubic', align_corners=True)
                out_tensor[i,...,rs:re, cs:ce] = tensor
            if name == 'mask':
                out_tensor = (out_tensor>0.5) * 1.
            elif name == 'normal_pred':
                out_tensor = F.normalize(out_tensor) * data['mask'].detach()
            elif name == 'material_est_pred':
                out_tensor = out_tensor * data['mask'].detach()
            else:
                if t_size_num == 4:
                    out_tensor = out_tensor.clamp(0, 1) * data['mask'].detach()
                elif t_size_num == 5:
                    out_tensor = out_tensor.clamp(0, 1) * data['mask'][:, None].detach()
                
            data[name] = out_tensor

def get_model_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
