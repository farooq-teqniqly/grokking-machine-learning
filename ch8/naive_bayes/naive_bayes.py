import pandas as pd
import numpy as np
from pandas import DataFrame


def _process_email(text: str):
    return list(set(text.lower().split()))


def create_model(emails: DataFrame) -> dict:
    model = {}

    for index, email in emails.iterrows():
        for word in email["words"]:
            if word not in model:
                model[word] = {"spam": 1, "ham": 1}
            if word in model:
                if email["spam"]:
                    model[word]["spam"] += 1
                else:
                    model[word]["ham"] += 1

            p_spam = model[word]["spam"] / (model[word]["spam"] + model[word]["ham"])
            model[word]["p_spam"] = round(p_spam, 2)
            model[word]["p_ham"] = round(1 - p_spam, 2)

    email_total = len(emails)
    model["email_total"] = email_total
    spam_count = sum(emails["spam"])
    model["spam_count"] = spam_count
    model["ham_count"] = email_total - spam_count

    return model


def predict_naive_bayes(email: str, model: dict):
    spams = [1.0]
    hams = [1.0]
    words = set(email.lower().split())

    for word in words:
        if word in model:
            spams.append(model[word]["p_spam"])
            hams.append(model[word]["p_ham"])

    product_spams = np.long(np.prod(spams) * model["spam_count"])
    product_hams = np.long(np.prod(hams) * model["ham_count"])

    return product_spams / (product_spams + product_hams)


def main():
    emails = pd.read_csv("emails.csv")
    emails["words"] = emails["text"].apply(_process_email)

    model = create_model(emails)
    print(predict_naive_bayes("Hi mom how are you", model))
    print(predict_naive_bayes("buy cheap lottery easy money now", model))
    print(
        predict_naive_bayes("meet me at the lobby of the hotel at nine am to receive your free lottery ticket", model))
    print(predict_naive_bayes("asdfgh", model))

    pass


if __name__ == "__main__":
    main()
