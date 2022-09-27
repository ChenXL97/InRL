import multiprocessing


def parallel_excute(pool_size, func, arg_list):
    pool = multiprocessing.Pool(processes=pool_size)
    pool_list = [pool.apply_async(func, args) for args in arg_list]
    result_list = [r.get() for r in pool_list]
    pool.close()
    pool.join()
    return result_list
