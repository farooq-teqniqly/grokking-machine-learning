import turicreate as tc

data = tc.SFrame("Hyderabad.csv")


def predict_house_price():
    model = tc.linear_regression.create(data, target="Price")
    house = tc.SFrame({"Area": [1000], "No. of Bedrooms": [3]})
    predicted_house_price = model.predict(house)
    print(predicted_house_price)


def predict_house_price_simple():
    simple_model = tc.linear_regression.create(data, features=["Area"], target="Price")
    coefficients = simple_model.coefficients
    print(coefficients)
    house = tc.SFrame({"Area": [1000]})
    predicted_house_price = simple_model.predict(house)
    print(predicted_house_price)


if __name__ == "__main__":
    predict_house_price()
    predict_house_price_simple()
