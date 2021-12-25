import pandas as pd
from sklearn.model_selection import train_test_split


def clean_up_data():
    raw_data = pd.read_csv("train.csv", index_col="PassengerId")
    clean_data = raw_data.drop("Cabin", axis=1)
    median_age = clean_data["Age"].median()
    clean_data["Age"] = clean_data["Age"].fillna(median_age)
    clean_data["Embarked"] = clean_data["Embarked"].fillna("U")
    clean_data.to_csv("titanic_data_clean.csv", index=None)


def preprocess_data():
    data = pd.read_csv("titanic_data_clean.csv")
    gender_cols = pd.get_dummies(data["Sex"], prefix="Sex")
    embarked_cols = pd.get_dummies(data["Pclass"], prefix="Pclass")
    data = pd.concat([data, gender_cols], axis=1)
    data = pd.concat([data, embarked_cols], axis=1)
    data = data.drop(["Sex", "Embarked", "Pclass"], axis=1)

    bins = list(range(0, 80, 10))
    ages_binned = pd.cut(data["Age"], bins)
    data["Categorized_age"] = ages_binned
    age_cols = pd.get_dummies(data["Categorized_age"], prefix="Categorized_age")
    data = pd.concat([data, age_cols], axis=1)
    data = data.drop(["Age", "Categorized_age", "Name", "Ticket"], axis=1)
    data.to_csv("titanic_data_preprocessed.csv", index=False)


def train_model(training_funcs: list):
    data = pd.read_csv("titanic_data_preprocessed.csv")
    features = data.drop(["Survived"], axis=1)
    labels = data["Survived"]

    features_train, features_validation_test, labels_train, labels_validation_test = train_test_split(
        features,
        labels,
        test_size=0.4)

    features_validation, features_test, labels_validation, labels_test = train_test_split(
        features_validation_test,
        labels_validation_test,
        test_size=0.5)

    for f in training_funcs:
        model = f(features_train, labels_train)
        print(f"{type(model)}: {model.score(features_validation, labels_validation)}")


def train_logistic_regression(training_set, training_set_labels):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(training_set, training_set_labels)

    return model


def train_decision_tree(training_set, training_set_labels):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    model.fit(training_set, training_set_labels)

    return model


def train_gradient_boosted_tree(training_set, training_set_labels):
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier()
    model.fit(training_set, training_set_labels)

    return model


def train_naive_bayes(training_set, training_set_labels):
    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()
    model.fit(training_set, training_set_labels)

    return model


if __name__ == '__main__':
    # clean_up_data()
    # preprocess_data()
    train_model([
        train_logistic_regression,
        train_decision_tree,
        train_gradient_boosted_tree,
        train_naive_bayes])
    pass
