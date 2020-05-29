import random
from typing import Tuple


def square_trick(
    price_per_room: int,
    base_price: int,
    num_rooms: int,
    price: int,
    learning_rate: float,
) -> Tuple[float, float]:
    predicted_price = base_price + (price_per_room * num_rooms)
    base_price += learning_rate * (price - predicted_price)
    price_per_room += learning_rate * num_rooms * (price - predicted_price)
    return price_per_room, base_price


def linear_regression(
    features: list, labels: list, learning_rate=0.01, epochs=1000
) -> Tuple[float, float]:
    price_per_room = random.random()
    base_price = random.random()

    for i in range(epochs):
        i = random.randint(0, len(features) - 1)
        num_rooms = features[i]
        price = labels[i]
        price_per_room, base_price = square_trick(
            base_price, price_per_room, num_rooms, price, learning_rate
        )

    return price_per_room, base_price
