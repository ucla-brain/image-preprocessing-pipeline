import argparse
import logging
import platform
import shutil
import subprocess
from json import dump
from pathlib import Path

import psutil

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

is_windows = platform.system() == "Windows"

def construct_cache_drive_folder_name(lambda_ex, lambda_em, brain_name=None):
    if brain_name:
        return f"cache_deconvolution_{brain_name}_Ex_{lambda_ex}_Em_{lambda_em}"
    else:
        return f"cache_deconvolution_Ex_{lambda_ex}_Em_{lambda_em}"

def find_matlab_executable():
    matlab_exec = "matlab.exe" if is_windows else "matlab"
    if shutil.which(matlab_exec):
        return matlab_exec
    possible_paths = [
        Path("C:/Program Files/MATLAB/R2023a/bin/matlab.exe"),
        Path("/usr/local/MATLAB/R2023a/bin/matlab"),
        Path("/opt/MATLAB/R2023a/bin/matlab")
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)
    raise RuntimeError("MATLAB executable not found. Add it to your PATH or update the script.")

def get_all_gpu_indices():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        indices = [int(line.strip()) + 1 for line in result.stdout.strip().splitlines()]
        return indices
    except FileNotFoundError:
        log.error("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in PATH.")
        return []
    except Exception as e:
        log.warning(f"GPU index detection failed: {e}")
        return []

def estimate_block_size_max(gpu_indices, num_workers, bytes_per_element=4, base_reserve_gb=0.1, per_worker_mib=160,
                            num_blocks_on_gpu=3):
    max_allowed = 2**31 - 10**6
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        all_memories = [int(x.strip()) for x in result.stdout.strip().splitlines()]
        selected_memories = [all_memories[i - 1] for i in gpu_indices if 0 <= i - 1 < len(all_memories)]
        if not selected_memories:
            return max_allowed

        min_vram_mib = min(selected_memories)
        usable_mib = min_vram_mib - base_reserve_gb * 1024 - num_workers * per_worker_mib
        if usable_mib <= 0:
            return max_allowed

        usable_bytes = usable_mib * 1024**2
        estimated = int(usable_bytes / bytes_per_element / num_blocks_on_gpu)
        return min(estimated, max_allowed)
    except Exception:
        log.warning("Could not estimate GPU memory, using safe default.")
        return max_allowed

def resolve_path(p):
    p = Path(p)
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    return p.resolve()

def validate_args(args):
    args.input = resolve_path(args.input)

    if args.lambda_ex not in [488, 561, 642] or args.lambda_em not in [525, 600, 690]:
        raise RuntimeError("Unsupported excitation/emission wavelength. Valid pairs: 488/525, 561/600, 642/690.")

    if [488, 561, 642].index(args.lambda_ex) != [525, 600, 690].index(args.lambda_em):
        raise RuntimeError(f"Ex {args.lambda_ex} and Em {args.lambda_em} do not match.")

    if len(args.sigma) != 3:
        raise ValueError("Sigma must be a triplet, e.g., --sigma 0.5 0.5 1.5")

    if len(args.filter_size) != 3:
        raise ValueError("Filter size must be a triplet, e.g., --filter_size 5 5 15")

def main():
    default_gpu_indices = get_all_gpu_indices()
    default_cores = psutil.cpu_count(logical=False)
    default_workers_per_gpu = (
        default_cores // len(default_gpu_indices)
        if len(default_gpu_indices) > 0 else 0
    )
    block_size_default = estimate_block_size_max(default_gpu_indices, default_workers_per_gpu * len(default_gpu_indices))

    parser = argparse.ArgumentParser(
        description='Python wrapper for MATLAB deconvolution.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--version', action='version', version='DeconvWrapper v1.2')

    # Required
    parser.add_argument('-i', '--input', type=Path, required=True,
        help='Path to the input image folder')
    parser.add_argument('-dxy', '--dxy', type=float, required=True,
        help='Lateral resolution in micrometers (μm)')
    parser.add_argument('-dz', '--dz', type=float, required=True,
        help='Axial resolution in micrometers (μm)')

    # Optional
    parser.add_argument('--brain_name', type=str, default=None,
        help='Optional brain name for cache path construction')
    parser.add_argument('--numit', '-it', type=int, default=10,
        help='Number of deconvolution iterations [1-50]')
    parser.add_argument('--lambda_ex', '-ex', type=int, default=488,
        help='Excitation wavelength (488, 561, 642)')
    parser.add_argument('--lambda_em', '-em', type=int, default=525,
        help='Emission wavelength (525, 600, 690)')
    parser.add_argument('--na', type=float, default=0.40,
        help='Numerical aperture of the objective lens')
    parser.add_argument('--rf', type=float, default=1.42,
        help='Refractive index of the sample medium')
    parser.add_argument('--fcyl', type=int, default=240,
        help='Focal length of the cylindrical lens (in mm)')
    parser.add_argument('--slitwidth', type=float, default=12.0,
        help='Slit width in millimeters')
    parser.add_argument('--lambda_damping', type=float, default=0.0,
        help='Damping parameter (0 = off)')
    parser.add_argument('--clipval', type=int, default=0,
        help='Clipping value (0 = disabled)')
    parser.add_argument('--stop_criterion', type=int, default=0,
        help='Early stopping criterion (0 = disabled)')
    parser.add_argument('--block_size_max', type=int, default=block_size_default,
        help='Max number of elements per GPU block (estimated from GPU memory)')
    parser.add_argument('--gpu-indices', type=int, nargs='+', default=default_gpu_indices,
        help='List of GPU device indices to use (e.g., 1 2). Default: all detected GPUs.')
    parser.add_argument('--gpu-workers-per-gpu', type=int, default=default_workers_per_gpu,
        help='Number of parallel workers per selected GPU')
    parser.add_argument('--cpu-workers', type=int, default=0,
        help='Number of CPU workers to use (0 disables CPU deconvolution)')
    parser.add_argument('--signal_amp', type=float, default=1.0,
        help='Signal amplification factor')
    parser.add_argument('--sigma', type=float, nargs=3, default=[0.5, 0.5, 1.5],
        help='3D Gaussian filter sigma in voxel unit (e.g., 0.5 0.5 1.5). Use 0 0 0 to disable filtering.')
    parser.add_argument('--filter_size', type=int, nargs=3, default=[5, 5, 15],
        help='Size of the 3D Gaussian filter kernel in voxel unit')
    parser.add_argument('--denoise_strength', type=int, default=1,
        help='Denoising strength (e.g., 1 to 255 for 8-bit images)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
        help='Disable resuming from previous cache (default: resume is enabled)')
    parser.set_defaults(resume=True)
    parser.add_argument('--flip', action='store_true',
        help='Flip output image vertically after deconvolution')
    parser.add_argument('--convert-to-8bit', action='store_true',
        help='Convert output to 8-bit (default keeps original bit depth, usually 16-bit)')
    parser.add_argument('--start_block', type=int, default=1,
        help='Starting block index for multi-GPU chunking')
    parser.add_argument('--dry-run', action='store_true',
        help='Print the MATLAB command and exit without executing it')

    args = parser.parse_args()

    # Re-estimate block size if user selected subset of GPUs and did not override block size
    user_specified_subset = (
        set(args.gpu_indices) != set(default_gpu_indices)
        and set(args.gpu_indices).issubset(set(default_gpu_indices))
    )
    user_overrode_block_size = args.block_size_max != block_size_default
    if user_specified_subset and not user_overrode_block_size:
        args.block_size_max = estimate_block_size_max(
            args.gpu_indices,
            args.gpu_workers_per_gpu * len(args.gpu_indices)
        )
        log.info(f"Re-estimated block_size_max: {args.block_size_max}")

    validate_args(args)

    gpu_worker_list = sum([[gpu] * args.gpu_workers_per_gpu for gpu in args.gpu_indices], [])
    cpu_worker_list = [0] * args.cpu_workers
    final_gpu_indices = gpu_worker_list + cpu_worker_list

    total_workers_requested = len(final_gpu_indices)
    total_cores = psutil.cpu_count(logical=False)
    if total_workers_requested < 1:
        raise RuntimeError("At least one GPU or CPU worker must be specified.")
    if total_workers_requested > total_cores:
        log.warning(f"You requested {total_workers_requested} workers but only have {total_cores} physical cores.")

    gpu_indices_str = ' '.join(str(i) for i in final_gpu_indices)
    sigma_str = ' '.join(str(i) for i in args.sigma)
    filter_size_str = ' '.join(str(i) for i in args.filter_size)
    cache_drive_folder = construct_cache_drive_folder_name(args.lambda_ex, args.lambda_em, args.brain_name)

    try:
        deconvolve_dir = Path(__file__).resolve().parent.as_posix()
    except NameError:
        deconvolve_dir = Path.cwd().as_posix()

    if args.dry_run:
        log.info(f"Estimated block_size_max: {args.block_size_max}")

    matlab_code = (
        f"addpath('{deconvolve_dir}'); "
        f"deconvolve('{args.input.as_posix()}', '{args.dxy * 1000}', '{args.dz * 1000}', "
        f"'{args.numit}', '{args.lambda_ex}', '{args.lambda_em}', '{cache_drive_folder}', "
        f"'{args.na}', '{args.rf}', '{args.fcyl}', '{args.slitwidth}', '{args.lambda_damping}', "
        f"'{args.clipval}', '{args.stop_criterion}', '{args.block_size_max}', "
        f"[{gpu_indices_str}], '{args.signal_amp}', [{sigma_str}], [{filter_size_str}], "
        f"'{args.denoise_strength}', '{int(args.resume)}', '{args.start_block}', "
        f"'{int(args.flip)}', '{int(args.convert_to_8bit)}', '{cache_drive_folder}'); exit;"
    )

    try:
        matlab_exec = find_matlab_executable()
    except RuntimeError as e:
        raise SystemExit(str(e))

    matlab_cmd = [matlab_exec, "-batch", matlab_code]

    log.info("MATLAB command:")
    log.info(' '.join(matlab_cmd) if is_windows else matlab_cmd)

    with open("deconvolution_config.json", "w") as f:
        dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    if args.dry_run:
        log.info("Dry run enabled. Command was not executed.")
        return

    try:
        log.info("Running MATLAB deconvolution...")
        subprocess.run(
            ' '.join(matlab_cmd) if is_windows else matlab_cmd,
            shell=is_windows,
            text=True,
            check=True
        )
        log.info("Deconvolution completed successfully.")
    except subprocess.CalledProcessError as e:
        log.error("MATLAB exited with error:")
        raise SystemExit(e)

if __name__ == "__main__":
    main()