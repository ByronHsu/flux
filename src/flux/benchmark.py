import os
import time
import gc
import argparse
import torch
from einops import rearrange
from PIL import Image, ExifTags
from torch.profiler import profile, record_function, ProfilerActivity

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline

from dataclasses import dataclass

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    name: str

def generate_image(opts, t5, clip, model, ae, torch_device, offload):
    x = get_noise(
        1,
        opts.height,
        opts.width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=opts.seed,
    )
    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(torch_device), clip.to(torch_device)
    inp = prepare(t5, clip, x, prompt=opts.prompt)
    timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(opts.name != "flux-schnell"))

    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)
    
    return x

@torch.inference_mode()
def benchmark_generation(args):
    if args.name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {args.name}, chose from {available}")

    torch_device = torch.device(args.device)
    if args.num_steps is None:
        args.num_steps = 4 if args.name == "flux-schnell" else 50

    height = 16 * (args.height // 16)
    width = 16 * (args.width // 16)

    t5 = load_t5(torch_device, max_length=256 if args.name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(args.name, device="cpu" if args.offload else torch_device)
    ae = load_ae(args.name, device="cpu" if args.offload else torch_device)

    t5 = torch.compile(t5, mode="max-autotune")
    clip = torch.compile(clip, mode="max-autotune")
    model = torch.compile(model, mode="max-autotune")
    ae = torch.compile(ae, mode="max-autotune")

    opts = SamplingOptions(
        prompt=args.prompt,
        width=width,
        height=height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        name=args.name
    )

    def run_generation():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return generate_image(opts, t5, clip, model, ae, torch_device, args.offload)

    # Warmup run
    print("Performing warmup run...")
    _ = run_generation()

    print(f"Benchmarking {args.num_runs} runs...")
    times = []
    memory_peaks = []

    with profile(
        schedule=torch.profiler.schedule(wait=0, warmup=args.profiler_step, active=2, repeat=1, skip_first=0),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler('logs/profiler_output'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(args.num_runs):
            prof.step()
            start_time = time.time()
            x = run_generation()
            end_time = time.time()

            generation_time = end_time - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MiB
            times.append(generation_time)
            memory_peaks.append(peak_memory)

            print(f"Run {i+1}: Time = {generation_time:.2f}s, Peak Memory = {peak_memory:.2f} MiB")

            # Save the generated image
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            img.save(f"benchmark_output_{i+1}.jpg", quality=95, subsampling=0)

    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_peaks) / len(memory_peaks)
    max_memory = max(memory_peaks)

    print(f"\nResults for {args.name}:")
    print(f"Average Generation Time: {avg_time:.2f} seconds")
    print(f"Average Peak Memory Usage: {avg_memory:.2f} MiB")
    print(f"Maximum Peak Memory Usage: {max_memory:.2f} MiB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Flux Image Generation")
    parser.add_argument("--name", type=str, default="flux-schnell", help="Name of the model to load")
    parser.add_argument("--width", type=int, default=1360, help="Width of the sample in pixels (should be a multiple of 16)")
    parser.add_argument("--height", type=int, default=768, help="Height of the sample in pixels (should be a multiple of 16)")
    parser.add_argument("--seed", type=int, default=42, help="Set a seed for sampling")
    parser.add_argument("--prompt", type=str, default="a photo of a forest with mist swirling around the tree trunks", help="Prompt used for sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Pytorch device")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of sampling steps")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance value used for guidance distillation")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--profiler_step", type=int, default=100, help="Which iteration to run the profiler on (0-indexed)")

    args = parser.parse_args()
    benchmark_generation(args)