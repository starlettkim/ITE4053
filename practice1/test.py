from binary_classifier import *
import time

if __name__ == '__main__':
    NUM_TEST = 100

    # Time comparison
    time_start = time.time()
    train_acc = 0
    test_acc = 0
    for test in range(NUM_TEST):
        result = train_binary_classifier(1000, 100, 100, 1e-1)
        train_acc += result['train_acc'] / NUM_TEST
        test_acc += result['test_acc'] / NUM_TEST
    time_elapsed = time.time() - time_start
    print('Total time elapsed: %fs' % time_elapsed)
    print('Average time elapsed: %fs' % (time_elapsed / NUM_TEST))
    print('Train acc: %f' % train_acc)
    print('Test acc: %f' % test_acc)
