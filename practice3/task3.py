import test
import nn

model = nn.models.Sequential([
            nn.layers.Dense(2, 3, nn.activations.Sigmoid),
            nn.layers.Dense(3, 1, nn.activations.Sigmoid)
        ])

if __name__ == '__main__':
    results = test.test_models([model], 1, 1000, 100)
    for item in results[0].items():
        print('%s: %f' % item)
    print()
