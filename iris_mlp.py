import pandas as pd
import numpy as np

from mlp import nn, metrics


def load_data(data_filepath):
    df = pd.read_csv(data_filepath)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    data = np.array(df)
    return data


def data_preprocess(data):
    iris_data = np.concatenate([data, np.zeros((150, 2))], axis=1)
    for i in range(len(iris_data)):
        if 'setosa' in iris_data[i]:
            iris_data[i, -3] = 1.0
        elif 'versicolor' in iris_data[i]:
            iris_data[i, -3] = 0.0
            iris_data[i, -2] = 1.0
        elif 'virginica' in iris_data[i]:
            iris_data[i, -3] = 0.0
            iris_data[i, -1] = 1.0
    return iris_data


def decoder(predicted, label_meaning):
    indices = np.argmax(predicted, axis=1)
    result = None
    for i in range(len(indices)):
        result = np.row_stack((result, label_meaning[int(indices[i])]))
    return result[1:]


def accuracy(model, feature, label_meaning, raw_data):
    predicted = decoder(model.predict(feature), label_meaning)[:, 0]
    predict_zip = list(zip(predicted, raw_data[:, 4]))
    t = 0
    f = 0
    for z in predict_zip:
        if z[0] != z[1]:
            f += 1
        elif z[0] == z[1]:
            t += 1
    acc = t / (t + f)
    print('accuracy: ', acc)


def learn(model, params_filepath, data_train, data_val, batch_size):
    nn.load(model, params_filepath)
    model.learn(data_train, data_val, batch_size, 500)
    nn.save_model(model, params_filepath)


def main():
    feature_size = 4  # 特征数量
    hidden_size = 5  # 隐含层神经元数量
    label_size = 3  # 标签数量
    batch_size = 10  # batch的大小
    # dataset_size = 100  # 数据集大小
    label_meaning = ['setosa', 'versicolor', 'virginica']
    data_filepath = './DataSets/iris.csv'
    params_filepath = './CheckPoints/iris_params.csv'
    iris_model = nn.TwoLayerNet(feature_size, hidden_size, label_size)

    iris_data = load_data(data_filepath)
    main_data = nn.DataSet(data_preprocess(iris_data))

    feature_1, label_1 = nn.DataLoader(main_data, batch_size).load(feature_size, label_size)
    data_train, data_val = main_data.split()
    # learn(iris_model, params_filepath, data_train, data_val, batch_size)

    accuracy(iris_model, feature_1, label_meaning, iris_data)


if __name__ == '__main__':
    main()
