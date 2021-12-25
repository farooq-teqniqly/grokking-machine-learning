import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

dataset = pd.DataFrame({
    "x_0": [7, 3, 2, 1, 2, 4, 1, 8, 6, 7, 8, 9],
    "x_1": [1, 2, 3, 5, 6, 7, 9, 10, 5, 8, 4, 6],
    "y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})


def main():
    features = dataset[["x_0", "x_1"]]
    labels = dataset["y"]

    decision_tree = DecisionTreeClassifier()
    classifier = decision_tree.fit(features, labels)
    classifier_as_text = export_text(classifier, feature_names=["x_0", "x_1"])
    print(classifier_as_text)


if __name__ == '__main__':
    main()
