import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from time import time
import argparse
import logging
import os
import random

from models.dsdit_sr import DSDiT
from models.model_wrappers import SDVAE, SD3LatentFormat, ModelSamplingDiscreteFlow
from dataset_sr import SuperResolutionDataset, get_sr_transforms
from safetensors import safe_open


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """End DDP training."""
    dist.destroy_process_group()


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_into(ckpt, model, prefix, device, dtype=None):
    """Load weights from safetensors into pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix):].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}': {e}")
                raise e


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,))
        u = torch.nn.functional.sigmoid(u)
    else:  # uniform
        u = torch.rand(size=(batch_size,))
    return u


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """Train DSDiT-RefSR model (end-to-end)."""
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup experiment folder
    experiment_dir = None
    checkpoint_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-dsdit"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Loading SD3 pretrained weights from: {args.pretrained_model}")
    else:
        logger = create_logger(None)

    # Read SD3 config
    with safe_open(args.pretrained_model, framework="pt", device="cpu") as f:
        patch_size = f.get_tensor("model.diffusion_model.x_embedder.proj.weight").shape[2]
        depth = f.get_tensor("model.diffusion_model.x_embedder.proj.weight").shape[0] // 64
        num_patches = f.get_tensor("model.diffusion_model.pos_embed").shape[1]

    pos_embed_max_size = round(num_patches ** 0.5)

    # Create model
    model = DSDiT(
        input_size=None,
        pos_embed_scaling_factor=None,
        pos_embed_offset=None,
        pos_embed_max_size=pos_embed_max_size,
        patch_size=patch_size,
        in_channels=16,
        depth=depth,
        num_patches=num_patches,
        qk_norm=None,
        device=device,
        dtype=torch.bfloat16,
        verbose=(rank == 0),
    )

    # Load SD3 weights
    with safe_open(args.pretrained_model, framework="pt", device="cpu") as f:
        sd3_state_dict = {}
        for key in f.keys():
            if key.startswith("model.diffusion_model."):
                model_key = key.replace("model.diffusion_model.", "")
                tensor = f.get_tensor(key).to(device=device, dtype=torch.bfloat16)
                sd3_state_dict[model_key] = tensor

        # Initialize from SD3
        weight_log_file = f"{experiment_dir}/weight_initialization.log" if rank == 0 else None
        model._initialize_from_sd3(sd3_state_dict, log_file=weight_log_file)

    if rank == 0:
        logger.info("Loaded SD3 weights with DSDiT initialization")
        logger.info(f"Weight initialization log: {experiment_dir}/weight_initialization.log")

    # Setup model sampling
    model_sampling = ModelSamplingDiscreteFlow(shift=args.shift)

    model = DDP(model.to(device), device_ids=[rank])

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"Total Parameters: {num_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")

    # Load VAE
    if rank == 0:
        logger.info("Loading SD3 VAE...")
    vae = SDVAE(dtype=torch.float16, device=device)
    with safe_open(args.pretrained_model, framework="pt", device="cpu") as f:
        prefix = "first_stage_model." if any(k.startswith("first_stage_model.") for k in f.keys()) else ""
        load_into(f, vae, prefix, device, torch.float16)
    vae.eval()
    requires_grad(vae, False)

    latent_format = SD3LatentFormat()

    # Setup optimizer
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Setup data
    if rank == 0:
        logger.info(f"Loading dataset from:")
        logger.info(f"  HR:  {args.hr_dir}")
        logger.info(f"  LR:  {args.lr_dir}")
        logger.info(f"  Ref: {args.ref_dir}")

    transform = get_sr_transforms(image_size=args.image_size)
    dataset = SuperResolutionDataset(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        ref_dir=args.ref_dir,
        transform=transform,
        image_size=args.image_size,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if rank == 0:
        logger.info(f"Dataset: {len(dataset):,} image triplets")

    # Prepare model for training
    model.train()

    # Training loop
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    epoch = 0

    while train_steps < args.max_steps:
        sampler.set_epoch(epoch)

        for hr, lr, ref in loader:
            hr = hr.to(device)
            lr = lr.to(device)
            ref = ref.to(device)

            # Encode to latent space
            with torch.no_grad():
                hr_latent = vae.encode(hr)
                hr_latent = latent_format.process_in(hr_latent)

                lr_latent = vae.encode(lr)
                lr_latent = latent_format.process_in(lr_latent)

                ref_latent = vae.encode(ref)
                ref_latent = latent_format.process_in(ref_latent)

            # Sample timestep
            t = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=hr_latent.shape[0],
                logit_mean=0.0,
                logit_std=1.0,
            ).to(device)

            # Sample noise
            noise = torch.randn_like(hr_latent)

            # Compute sigma and noisy latent
            sigma = model_sampling.sigma(t * 1000)
            noisy_hr = model_sampling.noise_scaling(sigma, noise, hr_latent)

            # Get timestep for model input
            timestep = model_sampling.timestep(sigma).float()

            ref_scale = 0.0 if random.random() < args.ref_scale_drop_prob else 1.0

            pred_velocity = model(
                noisy_hr.to(torch.bfloat16),
                lr_latent.to(torch.bfloat16),
                ref_latent.to(torch.bfloat16),
                timestep,
                ref_scale=ref_scale,
            ).float()

            # Target velocity
            target_velocity = (noise - hr_latent).float()

            # MSE loss
            loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if rank == 0:
                    logger.info(f"(step={train_steps:07d}, epoch={epoch}) Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            is_final = train_steps >= args.max_steps
            if (train_steps % args.ckpt_every == 0 and train_steps > 0) or is_final:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if is_final:
                break

        epoch += 1

    if rank == 0:
        logger.info(f"Training finished: {train_steps} steps, {epoch + 1} epochs")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to SD3 safetensors")
    parser.add_argument("--hr_dir", type=str, required=True, help="Path to HR images")
    parser.add_argument("--lr_dir", type=str, required=True, help="Path to LR images")
    parser.add_argument("--ref_dir", type=str, required=True, help="Path to Reference images")
    parser.add_argument("--results_dir", type=str, default="/root/results-dsdit")

    # Training
    parser.add_argument("--max_steps", type=int, default=80000, help="Total number of training steps")
    parser.add_argument("--global_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--shift", type=float, default=3.0)

    # Data
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--ref_scale_drop_prob", type=float, default=0.0,
                        help="Probability of setting ref_scale=0 during training for Autoguidance")

    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=10000)
    parser.add_argument("--global_seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
