from binary_classifier import *
import time


def run_test(m, n, K, learning_rate, num_runs):
    test_result = {'w': np.zeros(DIM_X), 'b': 0, 'train_acc': 0, 'test_acc': 0, 'elapsed': -time.time()}
    for run in range(num_runs):
        cls_result = train_binary_classifier(m, n, K, learning_rate)
        test_result['train_acc'] += cls_result['train_acc']
        test_result['test_acc'] += cls_result['test_acc']
        test_result['w'] += cls_result['w']
        test_result['b'] += cls_result['b']
    test_result['elapsed'] += time.time()
    for key, val in test_result.items():
        test_result[key] = val / num_runs
    return test_result


def test_m(m_list, n, K, learning_rate, num_runs):
    test_results = []
    for m in m_list:
        cur_result = {'m': m}
        cur_result.update(run_test(m, n, K, learning_rate, num_runs))
        test_results.append(cur_result)
    return test_results


def test_K(m, n, K_list, learning_rate, num_runs):
    test_results = []
    for K in K_list:
        cur_result = {'K': K}
        cur_result.update(run_test(m, n, K, learning_rate, num_runs))
        test_results.append(cur_result)
    return test_results


def test_lr(m, n, K, lr_list, num_runs):
    test_results = []
    for lr in lr_list:
        cur_result = {'learning_rate': lr}
        cur_result.update(run_test(m, n, K, lr, num_runs))
        test_results.append(cur_result)
    return test_results


if __name__ == '__main__':

    m = 100
    n = 100
    K = 100
    num_run = 100
    lr = 1e-2

    # Test m
    print('Testing m')
    print('with n = %d, K = %d, runs per test = %d, learning rate = %e' % (n, K, num_run, lr))
    test_results = test_m([10, 100, 1000], n, K, lr, num_run)
    for i in test_results:
        print(i)
    print()

    # Test K
    print('Testing K')
    print('with n = %d, m = %d, runs per test = %d, learning rate = %e' % (n, m, num_run, lr))
    test_results = test_K(m, n, [10, 100, 1000], lr, num_run)
    for i in test_results:
        print(i)
    print()

    # Test learning rate
    print('Testing learning rate')
    print('with n = %d, m = %d, K = %d, runs per test %d' % (n, m, K, num_run))
    test_results = test_lr(m, n, K, [1e-6, 1e-4, 1e-2, 1e0, 1e1, 1e2], num_run)
    for i in test_results:
        print(i)
    print()

    print(run_test(1000, 100, 1000, 1, 100))