from binary_classifier import *
import time

if __name__ == '__main__':
    NUM_TEST = 100

    # Time comparison
    time_start = time.time()
    for test in range(NUM_TEST):
        train_binary_classifier(1000, 100, 100, 1e-1)
    time_elapsed = time.time() - time_start
    print('Total time elapsed: %fs' % time_elapsed)
    print('Average time elapsed: %fs' % (time_elapsed / NUM_TEST))

