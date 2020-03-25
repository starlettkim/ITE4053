from binary_classifier import *
import time

if __name__ == '__main__':
    NUM_RUN = 100
    LEARN_RATE = 1e-6
    print('Running tests...')
    print('# of runs per test: %d' % NUM_RUN)
    print('Learning rate: %e\n' % LEARN_RATE)

    for m, K in [(10, 100), (100, 100), (1000, 100)]:
        train_acc, test_acc = (0, 0)
        time_start = time.time()
        for test in range(NUM_RUN):
            result = train_binary_classifier(m, 100, K, 1e-7)
            train_acc += result['train_acc'] / NUM_RUN
            test_acc += result['test_acc'] / NUM_RUN
        time_elapsed = time.time() - time_start
        print('Testing (m, K) = (%d, %d)' % (m, K))
        print('w: ' + str(result['w'][0]))
        print('b: ' + str(result['b']))
        print('Train accuracy: %f%%' % train_acc)
        print('Test accuracy: %f%%' % test_acc)
        print('Time elapsed: %fs' % (time_elapsed / NUM_RUN))
        print()

    print('Done.')