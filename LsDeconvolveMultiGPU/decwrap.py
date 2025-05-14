import argparse
import subprocess
import platform
import shutil
from pathlib import Path
import psutil

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
    except Exception:
        return []

def estimate_block_size_max(gpu_indices, num_workers, bytes_per_element=4, base_reserve_gb=.5, per_worker_mib=156):
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
        estimated = int(usable_bytes / bytes_per_element)
        return min(estimated, max_allowed)
    except Exception:
        print("WARNING: Could not estimate GPU memory, using safe default.")
        return max_allowed

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

    if not (args.input.exists() and args.input.is_dir()):
        raise RuntimeError(f"Input path {args.input} does not exist or is not a folder")

    if args.lambda_ex not in [488, 561, 642] or args.lambda_em not in [525, 600, 690]:
        raise RuntimeError("Unsupported excitation or emission wavelength")

    if [488, 561, 642].index(args.lambda_ex) != [525, 600, 690].index(args.lambda_em):
        raise RuntimeError(f"Ex {args.lambda_ex} and Em {args.lambda_em} do not match.")

    gpu_worker_list = sum([[gpu] * args.gpu_workers_per_gpu for gpu in args.gpu_indices], [])
    cpu_worker_list = [0] * args.cpu_workers
    final_gpu_indices = gpu_worker_list + cpu_worker_list

    total_workers_requested = len(final_gpu_indices)
    total_cores = psutil.cpu_count(logical=False)
    if total_workers_requested < 1:
        raise RuntimeError("At least one GPU or CPU worker must be specified.")
    if total_workers_requested > total_cores:
        print(f"WARNING: You requested {total_workers_requested} workers but only have {total_cores} physical cores.")

    gpu_indices_str = ' '.join(str(i) for i in final_gpu_indices)
    sigma_str = ' '.join(str(i) for i in args.sigma)
    filter_size_str = ' '.join(str(i) for i in args.filter_size)
    cache_drive_folder = construct_cache_drive_folder_name(args.lambda_ex, args.lambda_em, args.brain_name)
    deconvolve_dir = Path(__file__).resolve().parent.as_posix()

    if args.dry_run:
        print(f"Estimated block_size_max: {args.block_size_max}")

    matlab_code = (
        f"addpath('{deconvolve_dir}'); "
        f"deconvolve('{args.input.resolve().as_posix()}', '{args.dxy * 1000}', '{args.dz * 1000}', "
        f"'{args.numit}', '{args.lambda_ex}', '{args.lambda_em}', '{cache_drive_folder}', "
        f"'{args.na}', '{args.rf}', '{args.fcyl}', '{args.slitwidth}', '{args.lambda_damping}', "
        f"'{args.clipval}', '{args.stop_criterion}', '{args.block_size_max}', "
        f"[{gpu_indices_str}], '{args.signal_amp}', [{sigma_str}], [{filter_size_str}], "
        f"'{args.denoise_strength}', '{int(args.resume)}', '{args.start_block}', "
        f"'{int(args.flip)}', '{int(args.convert_to_8bit)}', '{cache_drive_folder}'); exit;"
    )

    matlab_exec = find_matlab_executable()
    matlab_cmd = [matlab_exec, "-nodisplay", "-r", matlab_code]

    print("MATLAB command:")
    print(' '.join(matlab_cmd) if is_windows else matlab_cmd)

    if args.dry_run:
        print("Dry run enabled. Command was not executed.")
        return

    try:
        print("Running MATLAB deconvolution...")
        result = subprocess.run(
            ' '.join(matlab_cmd) if is_windows else matlab_cmd,
            shell=is_windows,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            text=True
        )
        print("MATLAB STDOUT:\n", result.stdout)
        print("MATLAB STDERR:\n", result.stderr)
        result.check_returncode()
        print("Deconvolution completed successfully.")
    except subprocess.CalledProcessError as e:
        print("MATLAB exited with error:")
        print(e.stderr)
        exit(1)

if __name__ == "__main__":
    main()
