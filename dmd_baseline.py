from data_loader import load_dataset
import numpy as np
import os
import pydmd
import matplotlib.pyplot as plt

def main():

    ds = "unity_ident"
    extended = False

    # dataset = [[load_dataset(ds)[0][0]]]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dataset = load_dataset(ds)

    train_s = dataset[0].reshape((-1, *dataset[0].shape[2:])).T
    train_s = encode_states(train_s, extended)
    test_s = dataset[1].reshape((-1, *dataset[1].shape[2:])).T
    test_s = encode_states(test_s, extended)

    dmd = pydmd.DMD(20, 20, exact=True, opt=True)
    dmd.fit(train_s)


    plot_dims = [4, 5, 6]
    # plot_dims = [7, 8, 9]
    # time = 4828
    time = 200

    step_size = 40
    step_size += 1
    predictions = np.zeros((test_s[:, 0].shape[0], time))
    predictions[:, :step_size] = test_s[:, :step_size]
    for i in range(step_size, time):
        predictions[:, i] = dmd.predict(predictions[:, i-step_size:i-1])[:, 0]
        predictions[:4, i] = test_s[:4, i]


    predictions = decode_states(predictions, extended)

    # for i, traj in enumerate(dmd.predict(train_s[:time]).T[plot_dims]):
    for i, traj in enumerate(predictions[plot_dims]):
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(range(time), traj[:time], lw=1.5, linestyle='dashed')
        ax.plot(range(time), test_s.T[:, plot_dims[i]][:time], lw=1)

    plt.show()


def split_dataset(dataset, num_control):
    return dataset
    # return dataset[:, num_control:], dataset[:, :num_control]


def encode_states(dataset, extended):
    if not extended:
        return dataset

    sq = np.multiply(dataset, dataset)
    cub = np.multiply(sq, dataset)
    sin = np.sin(dataset)
    cos = np.cos(dataset)
    
    return np.vstack((dataset, sq, cub, sin, cos))



def decode_states(dataset, extended):
    if not extended:
        return dataset
    
    return dataset[:16, :]

if __name__ == "__main__":
    main()