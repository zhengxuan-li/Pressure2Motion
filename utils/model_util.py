from model.camdm import CAMDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

import torch
from os.path import join as pjoin

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("unexpected_keys: ", unexpected_keys)
    # assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):

    model = CAMDM(**get_model_args(args, data))
    # 1. Only rank 0 loads the checkpoint from disk
    if args.rank == 0:
        print('Loading pressure encoder on rank 0...')
        checkpoint_path = pjoin('./checkpoints/MPL/stage1_8gpu_8layers_root/model', 'latest.tar')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.input_cond_block.pressure_stage1.load_state_dict(checkpoint['encoder'])
    # 2. Move the model to the correct GPU on ALL processes BEFORE broadcasting
    model.to(args.device)
    # 2. Synchronize all processes to ensure rank 0 has finished loading
    if args.distributed:
        torch.distributed.barrier()

    # --- Freeze parameters on ALL processes ---
    print(f'Freezing parameters on rank {args.rank}...')
    for param in model.input_cond_block.pressure_stage1.parameters():
        param.requires_grad = False

    for param in model.input_process.parameters():
        param.requires_grad = False
    for param in model.sequence_pos_encoder.parameters():
        param.requires_grad = False
    for param in model.seqTransEncoder.parameters():
        param.requires_grad = False
    for param in model.embed_timestep.parameters():
        param.requires_grad = False
    for param in model.embed_text.parameters():
        param.requires_grad = False
    for param in model.output_process.parameters():
        param.requires_grad = False
    
    # ==== 检查哪些参数可训练 ====
    if args.rank == 0:
        print("\n=== Trainable Parameters ===")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"[Trainable] {name}")
            else:
                print(f"[Frozen] {name}")


    diffusion = create_gaussian_diffusion(args)
    
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    data_rep = 'hml_vec'
    njoints = 263
    nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        dataset=args.dataset
    )