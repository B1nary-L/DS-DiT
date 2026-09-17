import torch
import torch.distributed as dist
import argparse
import os
import math
import glob
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from safetensors import safe_open
from tqdm import tqdm
from models.dsdit_sr import DSDiT
from models.model_wrappers import SDVAE, SD3LatentFormat, ModelSamplingDiscreteFlow
from models.samplers import sample_euler, sample_dpmpp_2m


def load_vae(pretrained_model, device):
    """Load SD3 VAE."""
    vae = SDVAE(dtype=torch.float16, device=device)
    with safe_open(pretrained_model, framework="pt", device="cpu") as f:
        prefix = "first_stage_model." if any(k.startswith("first_stage_model.") for k in f.keys()) else ""
        for key in f.keys():
            if key.startswith(prefix) and not key.startswith("loss."):
                path = key[len(prefix):].split(".")
                obj = vae
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
                    tensor = f.get_tensor(key).to(device=device, dtype=torch.float16)
                    obj.requires_grad_(False)
                    obj.set_(tensor)
                except:
                    pass
    vae.eval()
    return vae


def load_image(image_path, image_size):
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


class DSDiTSampler:
    def __init__(self, model, model_sampling, ref_scale=1.0,
                 autog=False, ref_scale_weak=0.0, guidance=1.1):
        self.model = model
        self.model_sampling = model_sampling
        self.ref_scale = ref_scale
        self.autog = autog
        self.ref_scale_weak = ref_scale_weak
        self.guidance = guidance

    def __call__(self, hr_noisy, lr, ref, sigma):
        dtype = self.model.dtype
        timestep = self.model_sampling.timestep(sigma).float()

        hr_noisy_in = hr_noisy.to(dtype)
        lr_in = lr.to(dtype)
        ref_in = ref.to(dtype)

        vel = self.model(
            hr_noisy_in, lr_in, ref_in, timestep,
            ref_scale=self.ref_scale,
        ).float()

        if self.autog:
            vel_weak = self.model(
                hr_noisy_in, lr_in, ref_in, timestep,
                ref_scale=self.ref_scale_weak,
            ).float()
            vel = vel_weak + self.guidance * (vel - vel_weak)

        denoised = self.model_sampling.calculate_denoised(sigma, vel, hr_noisy)
        return denoised


def infer_sr(args):
    assert torch.cuda.is_available(), "Inference requires at least one GPU"
    torch.set_grad_enabled(False)

    # Setup DDP if enabled
    if args.use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        if rank == 0:
            print(f"Starting DDP with {dist.get_world_size()} GPUs")
    else:
        rank = 0
        device = args.device
        torch.manual_seed(args.seed)

    if rank == 0:
        print("=" * 60)
        print("Loading DSDiT-RefSR model...")
        print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=lambda storage, loc: storage, weights_only=False)

    # SD3-Medium standard configuration
    patch_size = 2
    depth = 24
    num_patches = 36864  # SD3 Medium: 192x192
    pos_embed_max_size = round(math.sqrt(num_patches))  # 192
    qk_norm = None
    rmsnorm = False

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
        qk_norm=qk_norm,
        rmsnorm=rmsnorm,
        device=device,
        dtype=torch.bfloat16,
        verbose=False,
    )

    # Load weights
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        if rank == 0:
            print("Loaded model weights")
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    # Load VAE
    if rank == 0:
        print("Loading VAE...")
    vae = load_vae(args.pretrained_model, device)
    latent_format = SD3LatentFormat()
    if rank == 0:
        print("VAE loaded")

    # Setup model sampling
    model_sampling = ModelSamplingDiscreteFlow(shift=args.shift)

    # Build timestep schedule
    start = model_sampling.timestep(model_sampling.sigma_max)
    end = model_sampling.timestep(model_sampling.sigma_min)
    timesteps = torch.linspace(start, end, args.steps, device=device)

    # Collect LR and Ref images
    lr_images = sorted(glob.glob(os.path.join(args.lr_dir, "*.*")))
    lr_images = [f for f in lr_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    ref_images = sorted(glob.glob(os.path.join(args.ref_dir, "*.*")))
    ref_images = [f for f in ref_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if rank == 0:
        print(f"Found {len(lr_images)} LR images and {len(ref_images)} Ref images")

    # Ensure same number of LR and Ref images
    assert len(lr_images) == len(ref_images), \
        f"Number of LR ({len(lr_images)}) and Ref ({len(ref_images)}) images must match!"

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.use_ddp:
        dist.barrier()

    # Split work among GPUs if using DDP
    if args.use_ddp:
        images_per_gpu = len(lr_images) // dist.get_world_size()
        start_idx = rank * images_per_gpu
        end_idx = start_idx + images_per_gpu if rank < dist.get_world_size() - 1 else len(lr_images)
        my_lr_images = lr_images[start_idx:end_idx]
        my_ref_images = ref_images[start_idx:end_idx]
        if rank == 0:
            print(f"Each GPU processing ~{images_per_gpu} images")
    else:
        my_lr_images = lr_images
        my_ref_images = ref_images

    # Select sampler function
    if args.sampler == "euler":
        sampler_fn = sample_euler
    elif args.sampler == "dpmpp_2m":
        sampler_fn = sample_dpmpp_2m
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    # Create sampler wrapper once (outside loop)
    sampler_wrapper = DSDiTSampler(
        model,
        model_sampling,
        ref_scale=args.ref_scale,
        autog=args.autog,
        ref_scale_weak=args.ref_scale_weak,
        guidance=args.guidance,
    )

    if rank == 0:
        print("=" * 60)
        print(f"Sampling with {args.steps} steps using {args.sampler.upper()}")
        if args.autog:
            print(f"  autoguidance    : ON")
            print(f"  ref_scale_weak  : {args.ref_scale_weak}")
            print(f"  guidance (w)    : {args.guidance}")
        else:
            print(f"  autoguidance    : OFF")
        print("=" * 60)

    iterator = (
        tqdm(zip(my_lr_images, my_ref_images), desc=f"[Rank {rank}] Sampling", total=len(my_lr_images))
        if rank == 0 else zip(my_lr_images, my_ref_images)
    )

    for lr_path, ref_path in iterator:
        img_name = os.path.basename(lr_path)
        name = os.path.splitext(img_name)[0]

        try:
            # Load and encode LR and Ref images
            lr = load_image(lr_path, args.image_size).to(device)
            ref = load_image(ref_path, args.image_size).to(device)

            with torch.no_grad():
                lr_latent = vae.encode(lr)
                lr_latent = latent_format.process_in(lr_latent)

                ref_latent = vae.encode(ref)
                ref_latent = latent_format.process_in(ref_latent)

            # Initial noise
            latent_size = args.image_size // 8
            hr_noisy = torch.randn(1, 16, latent_size, latent_size, device=device)
            sigmas = torch.cat([model_sampling.sigma(timesteps), torch.zeros(1, device=device)])

            # Sample
            with torch.no_grad():
                hr_latent = sampler_fn(
                    sampler_wrapper,
                    hr_noisy,
                    lr_latent,
                    ref_latent,
                    sigmas,
                )

                # Decode
                hr_latent = latent_format.process_out(hr_latent)
                hr_image = vae.decode(hr_latent)

            # Save output
            output_path = os.path.join(args.output_dir, f"{name}.png")
            save_image(hr_image, output_path, normalize=True, value_range=(-1, 1))

        except Exception as e:
            if rank == 0:
                print(f"Error processing {img_name}: {str(e)}")
            continue

    # Wait for all processes to finish
    if args.use_ddp:
        dist.barrier()

    if rank == 0:
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")
        print("=" * 60)

    if args.use_ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DSDiT-RefSR Inference Script")

    # Model paths
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to SD3 safetensors (for VAE)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained DSDiT-RefSR checkpoint")

    # Input/Output (batch mode)
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory containing LR images")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory containing Ref images")
    parser.add_argument("--output_dir", type=str, default="output_dsdit", help="Output directory")

    # Model settings
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")

    # Sampling settings
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "dpmpp_2m"], help="Sampling algorithm")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--shift", type=float, default=3.0, help="Flow shift parameter")
    parser.add_argument("--ref_scale", type=float, default=1.0,
                        help="Scale factor for ref attention")

    # Autoguidance settings
    parser.add_argument("--autog", action="store_true",
                        help="Enable Autoguidance: two forward passes per step")
    parser.add_argument("--ref_scale_weak", type=float, default=0.0,
                        help="ref_scale for the weak forward pass (only used with --autog)")
    parser.add_argument("--guidance", type=float, default=1.1,
                        help="Autoguidance coefficient w (only used with --autog). w=1 equals plain sampling; w>1 extrapolates")

    # Multi-GPU settings
    parser.add_argument("--use_ddp", action="store_true", help="Use DDP for multi-GPU processing")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Device (only used when not using DDP)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (ignored when using DDP)")

    args = parser.parse_args()

    infer_sr(args)


if __name__ == "__main__":
    main()
