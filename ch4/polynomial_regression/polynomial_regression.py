import turicreate as tc

data = tc.SFrame("data.csv")

for i in range(2, 200):
    string = f"x^{str(i)}"
    data[string] = data["x"].apply(lambda x: x ** i)

train, test = data.random_split(0.8)

penalties = [
    {
        "L1": 0,
        "L2": 0
    },
    {
        "L1": 0.1,
        "L2": 0
    },
    {
        "L1": 0,
        "L2": 0.1
    }
]

for p in penalties:
    model = tc.linear_regression.create(train, target="x", l1_penalty=p["L1"], l2_penalty=p["L2"])
