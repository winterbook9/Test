import numpy as np
import sys

# Dealing with data
class LoadData():
    def __init__(self, infile, attr):
        self.infile = infile
        self.attr = attr

    # load dataset from files
    def load_data(self):
        dataset = np.loadtxt(self.infile, dtype=str, delimiter='\t', skiprows=1)
        return dataset

    # get needed attr
    def get_attr(self, col):
        dataset = self.load_data()
        column = []

        for row in dataset:
            column.append(row[col])
        return column

    # get values of the specific attr
    def get_value(self, col):
        dataset = self.load_data()
        label1 = dataset[0][col]

        # assume the values are binary
        for i in dataset:
            if i[col] != label1:
                label2 = i[col]
                break
        return label1, label2


# Decision Stump
class DecisionStump():
    def __init__(self, infile, outfile, attr):
        self.outfile = outfile
        self.attr = attr

        self.ld = LoadData(infile, attr)
        self.attrColumn = self.ld.get_attr(self.attr)
        self.dataset = self.ld.load_data()

    # majority vote
    def majority_vote(self, dataset):
        count1 = 0
        label = self.ld.get_value(-1)

        for row in dataset:
            if row[-1] == label[0]:
                count1 += 1
            else:
                continue
        count2 = len(dataset) - count1

        if count1 > count2:
            return label[0]
        else:
            return label[1]

    # train
    def train_model(self):
        dataset = self.dataset
        attrColumn = self.attrColumn
        label = self.ld.get_value(self.attr)
        data0 = []
        data1 = []

        for i,row in enumerate(dataset):
            if attrColumn[i] == label[0]:
                data0.append(row)
            else:
                data1.append(row)

        # transfer to corresponding labels
        vote0 = self.majority_vote(data0)
        vote1 = self.majority_vote(data1)
        return vote0, vote1

    # predict and generate labels
    def h(self, vote):
        predict_labels = []
        label = self.ld.get_value(self.attr)

        for row in self.dataset:
            if row[self.attr] == label[0]:
                predict_labels.append(vote[0])
            else:
                predict_labels.append(vote[1])
        predict_labels = np.array(predict_labels)

        # write data.labels
        np.savetxt(self.outfile, predict_labels, fmt='%s', delimiter='\n')
        return predict_labels

    # evaluate
    def evaluate(self):
        metric = Metrics(self.ld.get_attr(-1), self.h(self.train_model()))
        return metric.error_rate()


# Choose error rate as precision evaluation
class Metrics():
    def __init__(self, real_col, pred_col):
        self.real_col = real_col
        self.pred_col = pred_col

    def error_rate(self):
        count = 0

        for i in range(len(self.real_col)):
            if self.real_col[i] != self.pred_col[i]:
                count += 1
        
        error_rate = count / len(self.real_col)
        return error_rate


# Main
if __name__ == '__main__':
    train_infile = sys.argv[1]
    test_infile = sys.argv[2]
    attr = sys.argv[3]
    train_outfile = sys.argv[4]
    test_outfile = sys.argv[5]
    metrics_out = sys.argv[6]

    # train part
    train_ds = DecisionStump(train_infile, train_outfile, int(attr))
    train_train_model = train_ds.train_model()
    train_predict = train_ds.h(train_train_model)
    train_error_rate = train_ds.evaluate()

    # test part
    test_ds = DecisionStump(test_infile, test_outfile, int(attr))
    test_train_model = test_ds.train_model()
    test_predict = test_ds.h(test_train_model)
    test_error_rate = test_ds.evaluate()

    # write metrics.txt
    with open(metrics_out, 'w') as f:
        f.writelines("error(train): {}\n".format(train_error_rate))
        f.writelines("error(test): {}".format(test_error_rate))