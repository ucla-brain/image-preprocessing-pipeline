import argparse
import ctypes.util
import logging
import os
import platform
import signal
import shutil
import subprocess
from json import dump
from pathlib import Path
from subprocess import CalledProcessError
from typing import Optional

import psutil

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

is_windows = platform.system() == "Windows"
is_linux = platform.system() == "Linux"


def find_matlab_executable():
    matlab_exec = "matlab.exe" if is_windows else "matlab"
    if shutil.which(matlab_exec):
        return matlab_exec
    possible_paths = [
        Path("C:/Program Files/MATLAB/R2025a/bin/matlab.exe"),
        Path("/usr/local/MATLAB/R2025a/bin/matlab"),
        Path("/opt/MATLAB/R2025a/bin/matlab"),
        Path("C:/Program Files/MATLAB/R2023a/bin/matlab.exe"),
        Path("/usr/local/MATLAB/R2023a/bin/matlab"),
        Path("/opt/MATLAB/R2023a/bin/matlab")
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)
    raise RuntimeError("MATLAB executable not found. Add it to your PATH or update the script.")


def find_allocator(libname: str) -> Optional[str]:
    candidates = [
        f"/usr/lib/x86_64-linux-gnu/lib{libname}.so",
        f"/usr/lib/lib{libname}.so",
        f"/lib/x86_64-linux-gnu/lib{libname}.so",
        f"/usr/local/lib/lib{libname}.so",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    try:
        output = subprocess.check_output(["ldconfig", "-p"], text=True)
        for line in output.splitlines():
            if libname in line:
                parts = line.strip().split(" => ")
                if len(parts) == 2 and Path(parts[1]).exists():
                    return parts[1]
    except Exception as e:
        log.warning(f"Failed to run ldconfig for {libname}: {e}")

    try:
        lib = ctypes.util.find_library(libname)
        if lib:
            for directory in ["/usr/lib", "/usr/local/lib", "/lib", "/lib64"]:
                candidate_path = Path(directory) / lib
                if candidate_path.exists():
                    return str(candidate_path)
    except Exception as e:
        log.warning(f"ctypes.util.find_library failed for {libname}: {e}")

    return None


def get_all_gpu_indices():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        # MATLAB uses 1-based indexing for GPU IDs, so we offset the zero-based output
        indices = [int(line.strip()) + 1 for line in result.stdout.strip().splitlines()]
        return indices
    except FileNotFoundError:
        log.error("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in PATH.")
        return []
    except Exception as e:
        log.warning(f"GPU index detection failed: {e}")
        return []


def estimate_block_size_max(gpu_indices, workers_per_gpu,
                            bytes_per_element=4, base_reserve_gb=0.75, per_worker_mib=844, num_blocks_on_gpu=2):
    max_allowed = 2 ** 31 - 1
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        all_memories = [int(x.strip()) for x in result.stdout.strip().splitlines()]
        selected_memories = [all_memories[i - 1] for i in gpu_indices if 0 <= i - 1 < len(all_memories)]
        if not selected_memories:
            log.warning("No matching GPU indices found in nvidia-smi output.")
            return max_allowed

        min_vram_mib = min(selected_memories)
        usable_mib = min_vram_mib - base_reserve_gb * 1024 - workers_per_gpu * per_worker_mib
        if usable_mib <= 0:
            log.warning(f"Estimated usable VRAM ({usable_mib:.1f} MiB) is too low. Falling back to max_allowed.")
            return max_allowed

        usable_bytes = usable_mib * 1024 ** 2
        estimated = int(usable_bytes / bytes_per_element / num_blocks_on_gpu)
        return min(estimated, max_allowed)

    except (CalledProcessError, FileNotFoundError) as e:
        log.warning(f"nvidia-smi failed: {e}. Using safe default.")
    except (ValueError, IndexError) as e:
        log.warning(f"Unexpected output from nvidia-smi: {e}. Using safe default.")

    return max_allowed

def resolve_path(p):
    p = Path(p)
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    return p.resolve()

def validate_args(args):
    args.input = resolve_path(args.input)

    # Supported excitation/emission wavelength pairs (nm)
    supported_pairs = [
        (350, 460),  # DAPI
        (405, 450),  # AmCyan
        (430, 470),  # CFP
        (458, 480),  # mCerulean
        (488, 525),  # GFP
        (514, 530),  # YFP
        (532, 555),  # TRITC
        (561, 600),  # mCherry
        (594, 620),  # Texas Red
        (633, 660),  # Cy5
        (642, 690),  # Alexa 647
        (680, 710),  # Cy7
    ]

    ex = getattr(args, 'lambda_ex', None)
    em = getattr(args, 'lambda_em', None)

    if ex is None or em is None:
        raise RuntimeError("Missing required arguments: lambda_ex and lambda_em")

    if (ex, em) not in supported_pairs:
        # Build a readable string of valid pairs
        pair_str = ', '.join(f"{e}/{m}" for e, m in supported_pairs)
        raise RuntimeError(
            f"Unsupported excitation/emission pair: {ex}/{em}. "
            f"Valid pairs are: {pair_str}"
        )

    if len(args.gaussian_sigma) != 3:
        raise ValueError("Gaussian sigma must be a triplet, e.g., --gaussian-sigma 0.5 0.5 1.5")

    if len(args.gaussian_filter_size) != 3:
        raise ValueError("Gaussian filter size must be a triplet, e.g., --gaussian-filter-size 5 5 15")

def main():
    default_gpu_indices = get_all_gpu_indices()
    default_cores = psutil.cpu_count(logical=False)
    default_workers_per_gpu = (
        default_cores // len(default_gpu_indices)
        if len(default_gpu_indices) > 0 else 0
    )
    default_workers_per_gpu = min(7, default_workers_per_gpu)
    block_size_default = estimate_block_size_max(default_gpu_indices, default_workers_per_gpu)

    parser = argparse.ArgumentParser(
        description='Python wrapper for MATLAB deconvolution.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--version', action='version', version='DeconvWrapper v1.5')

    # Required
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Path to the input image folder')
    parser.add_argument('-dxy', '--dxy', type=float, required=True,
                        help='Lateral resolution in micrometers (μm)')
    parser.add_argument('-dz', '--dz', type=float,
                        help='Axial resolution in micrometers (μm)')
    parser.add_argument('-ex', '--lambda-ex', type=int, required=True,
                        help='Excitation wavelength (488, 561, 642)')
    parser.add_argument('-em', '--lambda-em', type=int, required=True,
                        help='Emission wavelength (525, 600, 690)')

    # Optional
    parser.add_argument('--use-fft', action='store_true', default=False,
                        help='use FFT-based convolution, which is faster but uses more memory.')
    parser.add_argument('--adaptive-psf', action='store_true', default=False,
                        help='use Wiener method to adaptively update the PSF at every step.')
    parser.add_argument('--cache-drive', type=str, default=None,
                        help='Optional brain name for cache path construction')
    parser.add_argument('-it', '--numit', type=int, default=6,
                        help='Number of deconvolution iterations [1-50]')
    parser.add_argument('--na', type=float, default=0.40,
                        help='Numerical aperture of the objective lens')
    parser.add_argument('--rf', type=float, default=1.42,
                        help='Refractive index of the sample medium')
    parser.add_argument('--fcyl', type=int, default=240,
                        help='Focal length of the cylindrical lens (in mm)')
    parser.add_argument('--slitwidth', type=float, default=12.0,
                        help='Slit width in millimeters')
    parser.add_argument('--lambda-damping', type=float, default=0.0,
                        help='Tikhonov (L2) regularization weight in the range [0, 1]. '
                             'Applies only during blind deconvolution (enabled when --regularize-interval > 0). '
                             'Blends each RL update with a smoothed version of the image to suppress noise. '
                             'Set to 0 to disable. Suggested values:\n'
                             '- 1e-5 to 1e-3: Low noise, high SNR data\n'
                             '- 1e-3 to 1e-2: Typical data with moderate noise\n'
                             '- 1e-2 to 0.1: High noise or very ill-posed cases\n'
                             '- Use slightly higher values for fewer iterations, lower for many iterations\n'
                             '- 5e-3 to 5e-2 recommended for Weiner deconvolution')
    parser.add_argument('--clipval', type=float, default=99.99,
                        help='Clip the upper intensity of the entire image at the given percentile (default: 99.99). '
                             'The upper limit is computed as max(percentile(all_blocks, clipval)). '
                             'Set to 0 to disable clipping.')
    parser.add_argument('--stop-criterion', type=float, default=0,
                        help='Early stopping threshold as a percentage change in loss between iterations. '
                             'Training stops if the relative change falls below this value. '
                             'Set to 0 to disable early stopping.')
    parser.add_argument('--block-size-max', type=int, default=block_size_default,
                        help='Max number of elements per GPU block (estimated from GPU memory)')
    parser.add_argument('--gpu-indices', type=int, nargs='+', default=default_gpu_indices,
                        help='List of GPU device indices to use (e.g., 1 2). Default: all detected GPUs.')
    parser.add_argument('--gpu-workers-per-gpu', type=int, default=default_workers_per_gpu,  # default_workers_per_gpu
                        help='Number of parallel workers per selected GPU')
    parser.add_argument('--cpu-workers', type=int, default=0,
                        help='Number of CPU workers to use (0 disables CPU deconvolution)')
    parser.add_argument('--signal-amp', type=float, default=1.0,
                        help='Signal amplification factor')
    parser.add_argument('--gaussian-sigma', type=float, nargs=3, default=[0.5, 0.5, 2.5],
                        help='3D Gaussian filter sigma in voxel unit (e.g., 0.5 0.5 1.5). Use 0 0 0 to disable filtering.')
    parser.add_argument('--gaussian-filter-size', type=int, nargs=3, default=[13, 13, 25],
                        help='Size of the 3D Gaussian filter kernel in voxel unit')
    parser.add_argument('--denoise-strength', type=int, default=1,
                        help='Denoising strength (e.g., 1 to 255 for 8-bit images)')
    parser.add_argument('--destripe-sigma', type=float, default=0.0,
                        help='Standard deviation (sigma) of the destriping filter applied along the z-axis. '
                             'Set to 0 to disable destriping. '
                             'A value around 1000 is recommended for older cameras to remove stripe artifacts; '
                             'for most modern cameras, destriping is usually unnecessary.')
    parser.add_argument('--regularize-interval', type=int, default=3,
                        help='Apply a 3D Gaussian smoothing filter (σ=0.5) to the deconvolved volume every N iterations. '
                             'Set to 0 to disable both smoothing and blind deconvolution.')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Disable resuming from previous cache (default: resume is enabled)')
    parser.set_defaults(resume=True)
    parser.add_argument('--flip', action='store_true',
                        help='Flip output image vertically after deconvolution')
    parser.add_argument('--convert-to-8bit', action='store_true',
                        help='Convert output to 8-bit (default keeps original bit depth, usually 16-bit)')
    parser.add_argument('--convert-to-16bit', action='store_true',
                        help='Convert output to 16-bit (default keeps original bit depth, usually 16-bit)')
    parser.add_argument('--start-block', type=int, default=1,
                        help='Starting block index for multi-GPU or multi-computer processing (default: 1). '
                             'Intended for collaborative workflows where multiple machines access the same image via a shared network path. '
                             'Slave computers should use --start-block > 1. '
                             'The machine with --start-block=1 acts as the master, responsible for assembling slabs and saving the final image.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the MATLAB command and exit without executing it')
    parser.add_argument('--use-jemalloc', action='store_true', default=False,
                        help='Use jemalloc allocator (Linux only)')
    parser.add_argument('--use-tcmalloc', action='store_true', default=False,
                        help='Use tcmalloc allocator (Linux only)')

    args = parser.parse_args()

    # Re-estimate block size if user selected subset of GPUs and did not override block size
    user_specified_subset = (
            set(args.gpu_indices) != set(default_gpu_indices)
            and set(args.gpu_indices).issubset(set(default_gpu_indices))
    )
    user_overrode_block_size = args.block_size_max != block_size_default
    n_blocks_on_gpu = 2
    if args.use_fft:
        n_blocks_on_gpu = 5
        if args.adaptive_psf:
            n_blocks_on_gpu = 9
    if args.lambda_damping and not args.adaptive_psf:
        n_blocks_on_gpu += 1
    if user_specified_subset or not user_overrode_block_size:
        args.block_size_max = estimate_block_size_max(
            args.gpu_indices,
            args.gpu_workers_per_gpu,
            num_blocks_on_gpu=n_blocks_on_gpu
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
    gaussian_sigma_str = ' '.join(str(i) for i in args.gaussian_sigma)
    gaussian_filter_size_str = ' '.join(str(i) for i in args.gaussian_filter_size)
    cache_drive_folder = Path(args.input) / f"cache_deconvolution_Ex_{args.lambda_ex}_Em_{args.lambda_em}"
    if args.cache_drive:
        if not Path(args.cache_drive).exists():
            raise RuntimeError(f"Cache drive folder does not exist: {args.cache_drive}")
        cache_drive_folder = Path(args.cache_drive) / cache_drive_folder.name

    try:
        deconvolve_dir = Path(__file__).resolve().parent.as_posix()
    except NameError:
        deconvolve_dir = Path.cwd().as_posix()



    if args.convert_to_8bit and args.convert_to_16bit:
        raise RuntimeError("Cannot use both convert-to-8bit and convert-to-16bit simultaneously. Choose one.")

    matlab_exec = find_matlab_executable()

    if args.use_jemalloc and args.use_tcmalloc:
        raise RuntimeError("Cannot use both jemalloc and tcmalloc simultaneously. Choose one.")

    jemalloc_path = find_allocator("jemalloc") if is_linux and args.use_jemalloc else None
    tcmalloc_path = find_allocator("tcmalloc") if is_linux and args.use_tcmalloc else None

    if args.use_jemalloc and is_linux:
        if jemalloc_path:
            log.info(f"Using jemalloc allocator at: {jemalloc_path}")
        else:
            log.warning("jemalloc requested but not found. To install on Ubuntu: sudo apt install libjemalloc2.")

    if args.use_tcmalloc and is_linux:
        if tcmalloc_path:
            log.info(f"Using tcmalloc allocator at: {tcmalloc_path}")
        else:
            log.warning("tcmalloc requested but not found. To install on Ubuntu: sudo apt install google-perftools.")

    tmp_script_path = Path("deconv_batch_script.m")
    tmp_script_path.write_text(
        f"addpath('{deconvolve_dir}');\n"
        f"LsDeconv( ...\n"
        f"    convertCharsToStrings('{args.input.as_posix()}'), ...\n"
        f"    {args.dxy * 1000}, ...\n"
        f"    {args.dz * 1000}, ...\n"
        f"    {args.numit}, ...\n"
        f"    {args.na}, ...\n"
        f"    {args.rf}, ...\n"
        f"    {args.lambda_ex}, ...\n"
        f"    {args.lambda_em}, ...\n"
        f"    {args.fcyl}, ...\n"
        f"    {args.slitwidth}, ...\n"
        f"    {args.lambda_damping}, ...\n"
        f"    {args.clipval}, ...\n"
        f"    {args.stop_criterion}, ...\n"
        f"    {args.block_size_max}, ...\n"
        f"    [{gpu_indices_str}], ...\n"
        f"    {args.signal_amp}, ...\n"
        f"    [{gaussian_sigma_str}], ...\n"
        f"    [{gaussian_filter_size_str}], ...\n"
        f"    {args.denoise_strength}, ...\n"
        f"    {args.destripe_sigma}, ...\n"
        f"    {args.regularize_interval}, ...\n"
        f"    {int(args.resume)}, ...\n"
        f"    {args.start_block}, ...\n"
        f"    {int(args.flip)}, ...\n"
        f"    {'true' if args.convert_to_8bit else 'false'}, ...\n"
        f"    {'true' if args.convert_to_16bit else 'false'}, ...\n"
        f"    {'true' if args.use_fft else 'false'}, ...\n"
        f"    {'true' if args.adaptive_psf else 'false'}, ...\n"
        f"    convertCharsToStrings('{cache_drive_folder.as_posix()}') ...\n"
        f");\n"
    )
    # === Insert Linux-specific optimizations ===
    env = os.environ.copy()
    if not is_windows:
        # Set glibc tuning to reduce fragmentation
        env["MALLOC_ARENA_MAX"] = "2"  # keep fragmentation down with fewer arenas
        env["MALLOC_MMAP_THRESHOLD_"] = "134217728"  # 128 MB — prefer heap for big blocks
        env["MALLOC_TRIM_THRESHOLD_"] = "-1"  # never trim — hold on to memory
        if jemalloc_path:
            env["LD_PRELOAD"] = jemalloc_path
        if tcmalloc_path:
            env["LD_PRELOAD"] = tcmalloc_path

        # Use numactl for better memory distribution on NUMA systems
        # Wrap the matlab command with numactl only on Linux
        matlab_cmd = [
            "numactl", "--interleave=all",
            matlab_exec,
            "-batch",
            f"run('{tmp_script_path.stem}')"
        ]
    else:
        # Windows-compatible MATLAB command
        matlab_cmd = [
            matlab_exec,
            "-batch",
            f"run('{tmp_script_path.stem}')"
        ]

    log.info("MATLAB command:")
    log.info(' '.join(matlab_cmd))

    decon_path = args.input / "deconvolved"
    decon_path.mkdir(exist_ok=True)

    with open(decon_path / "deconvolution_config.json", "w") as f:
        dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    if args.dry_run:
        log.info("Dry run enabled. Command was not executed.")
        return

    proc = None
    try:
        log.info("Running MATLAB deconvolution...")
        proc = subprocess.Popen(
            ' '.join(matlab_cmd) if is_windows else matlab_cmd,
            shell=is_windows,
            env=env,  # Pass the modified environment
            text=True,
            preexec_fn=os.setsid if not is_windows else None
        )
        proc.wait()

        if proc.returncode != 0:
            log.error(f"MATLAB exited with error code {proc.returncode}.")
            raise SystemExit(f"MATLAB execution failed with exit code {proc.returncode}. Check logs for details.")

        log.info("Deconvolution completed successfully.")

    except KeyboardInterrupt:
        log.warning("Interrupted by user (Ctrl+C). Terminating MATLAB...")

        if proc and proc.poll() is None:
            killed = False
            try:
                log.info("Terminating MATLAB process tree...")
                if is_windows:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], shell=True)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                killed = True
            except Exception as e:
                log.warning(f"Failed to terminate MATLAB process: {e}")

            if not killed:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception as e:
                    log.warning(f"Fallback termination also failed: {e}")

        raise SystemExit("Execution interrupted by user.")

    finally:
        if not args.dry_run and tmp_script_path.exists():
            log.info("Cleaning up temporary MATLAB script...")
            tmp_script_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
