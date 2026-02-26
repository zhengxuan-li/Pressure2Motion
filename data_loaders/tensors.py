# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import torch
import numpy as np


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    pressurebatch = [b['pressure'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    pressure = collate_tensors(pressurebatch)
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = [b['obj_points']for b in notnone_batches]
        cond['y'].update({'obj_points': torch.as_tensor(obj_points)})
    
    if 'feet_hint' in notnone_batches[0] and notnone_batches[0]['feet_hint'] is not None:
        feet_hint = [b['feet_hint']for b in notnone_batches]
        feet_hint = np.array(feet_hint, dtype=np.float32)
        cond['y'].update({'feet_hint': torch.from_numpy(feet_hint)})

    if 'joints' in notnone_batches[0] and notnone_batches[0]['joints'] is not None:
        joints = [b['joints']for b in notnone_batches]
        joints = np.array(joints, dtype=np.float32)
        cond['y'].update({'joints': torch.from_numpy(joints)})

    # collate pressure
    if 'pressure' in notnone_batches[0]:
        # pressure = [b['pressure'] for b in notnone_batches]
        cond['y'].update({'pressure': pressure})
    
    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'hint': b[-1],
    } for b in batch]
    return collate(adapted_batch)

# an adapter to our collate func
def mpl_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[5].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[7],
        'lengths': b[6],
        'pressure': torch.tensor(b[4]).float(),  # [seqlen + 1, 160, 120],
        'feet_hint': b[-2],
        'joints': b[-1]
    } for b in batch]
    return collate(adapted_batch)

