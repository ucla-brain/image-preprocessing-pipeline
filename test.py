from multiprocessing import Pool, freeze_support
from time import sleep
import tqdm
import signal
from functools import wraps


def handle_ctrl_c(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global ctrl_c_entered
        if not ctrl_c_entered:
            signal.signal(signal.SIGINT, default_sigint_handler) # the default
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                ctrl_c_entered = True
                return KeyboardInterrupt()
            finally:
                signal.signal(signal.SIGINT, pool_ctrl_c_handler)
        else:
            return KeyboardInterrupt()
    return wrapper


@handle_ctrl_c
def slowly_square(i):
    sleep(1)
    return i*i


def pool_ctrl_c_handler(*args, **kwargs):
    global ctrl_c_entered
    ctrl_c_entered = True


def init_pool():
    # set global variable for each process in the pool:
    global ctrl_c_entered
    global default_sigint_handler
    ctrl_c_entered = False
    default_sigint_handler = signal.signal(signal.SIGINT, pool_ctrl_c_handler)


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with Pool(processes=61, initializer=init_pool) as pool:
        results = tqdm.tqdm(
            pool.map(slowly_square, range(100)),
            total=100,
            ascii=True
        )
        list(results)
        # if any(list(map(lambda x: isinstance(x, KeyboardInterrupt), results))):
        #     print('Ctrl-C was entered.')
        # else:
        #     print(results)


if __name__ == '__main__':
    freeze_support()
    ctrl_c_entered, default_sigint_handler = None, None
    main()
