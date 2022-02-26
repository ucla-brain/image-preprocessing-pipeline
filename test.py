from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pystripe.raw import raw_imread
from pathlib import Path


if __name__ == '__main__':
    freeze_support()
    with ProcessPoolExecutor(max_workers=61) as pool:
        inputs = [
            Path(r"Y:\SmartSPIM_Data\2022_02_21\20220221_17_24_07_SA210705_01_WholeBrain_LS_15x_1000z\Ex_561_Em_600\122520\122520_183950\007220.raw"),
            Path(r"Y:\SmartSPIM_Data\2022_02_21\20220221_17_24_07_SA210705_01_WholeBrain_LS_15x_1000z\Ex_561_Em_600\122520\122520_183950\007210.raw"),
        ]
        # result = list(
        #     pool.map(
        #         raw_imread,
        #         [
        #             Path(r"Y:\SmartSPIM_Data\2022_02_21\20220221_17_24_07_SA210705_01_WholeBrain_LS_15x_1000z\Ex_561_Em_600\122520\122520_183950\007220.raw"),
        #             Path(r"Y:\SmartSPIM_Data\2022_02_21\20220221_17_24_07_SA210705_01_WholeBrain_LS_15x_1000z\Ex_561_Em_600\122520\122520_183950\007210.raw"),
        #         ],
        #         timeout=10, chunksize=1, unordered=True
        #     )
        # )
        future_to_path = {pool.submit(raw_imread, path, {"shape": (1850, 1850), "dtype": "uint16"}): path for path in inputs}
        for future in as_completed(future_to_path, timeout=10):
            try:
                result = future.result()
            except BrokenProcessPool:
                path = future_to_path[future]
                print(f"timeout for path:\n\t{path}")
            else:
                print(result)

