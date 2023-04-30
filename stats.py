import pstats
import cProfile


def execution_speed_lib(func):
    """
    実行速度計測用のデコレータ
    """

    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        # 実行処理の計測
        pr.runcall(func, *args, **kwargs)

        stats = pstats.Stats(pr)
        stats.print_stats()

    return wrapper


@execution_speed_lib
def run():
    import test_cython


run()
