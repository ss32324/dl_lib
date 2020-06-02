import numpy as np

class OneHot:
    def __init__(self, labels):
        self.kind_of_labels = np.unique(np.asarray(labels))
        self.onehot_contrasts = np.eye(len(self.kind_of_labels))

    def encoding(self, labels):
        ### labels to one_hot_labels
        onehot_labels = []
        for label in np.asarray(labels):
            lab = self.kind_of_labels[np.where(self.kind_of_labels==label)]
            onehot_labels.append(self.onehot_contrasts[:,lab])
        return np.asarray(onehot_labels)

    def decoding(self, onehot_labels):
        ### onehot_label to label
        decode = np.argmax(onehot_labels, axis=1)
        decode = decode.reshape(decode.shape[0])
        return decode