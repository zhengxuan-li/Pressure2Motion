import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

import torch.distributed as dist
import torch


def init_distributed_mode(opt):

    # Initialize the distributed process group
    dist.init_process_group(backend="nccl")
    opt.local_rank = int(os.environ["LOCAL_RANK"])
    opt.gpu_id = [opt.local_rank]
    opt.rank = dist.get_rank()
    opt.world_size = dist.get_world_size()
    opt.device = torch.device(f"cuda:{opt.local_rank}")
    torch.cuda.set_device(opt.device)
    opt.distributed = True

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    init_distributed_mode(args)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    args_dict = vars(args).copy()
    if 'device' in args_dict:
        del args_dict['device']
        
    with open(args_path, 'w') as fw:
        json.dump(args_dict, fw, indent=4, sort_keys=True)

    print("creating data loader...")
    data = get_dataset_loader(opt=args, name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    if model is None:
        # Add a sanity check to get a clearer error message if the problem persists
        raise RuntimeError(f"create_model_and_diffusion returned None on rank {args.rank}.")

    if args.distributed:
        # Prepare model for DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    # model.to(dist_util.dev())
    model.module.rot2xyz.smpl_model.eval()

    if args.rank == 0:
        print('Total params: %.2fM' % (sum(p.numel() for p in model.module.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
