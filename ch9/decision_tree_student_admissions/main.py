import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def main():
    data = pd.read_csv("Admission_Predict.csv", index_col=0)
    data["Admitted"] = data["Chance of Admit "] >= 0.75
    data = data.drop("Chance of Admit ", axis=1)

    features = data.drop("Admitted", axis=1)
    labels = data["Admitted"]

    decision_tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=10,
        min_samples_split=10)

    decision_tree.fit(features, labels)

    feature_names = [c for c in features.columns]
    print(export_text(decision_tree, feature_names=feature_names))

    print(decision_tree.score(features, labels))

    print(decision_tree.predict([[320, 110, 3, 4.0, 3.5, 8.9, 0]]))


if __name__ == '__main__':
    main()
