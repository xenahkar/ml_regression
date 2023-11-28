from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


with open("best_model.pickle", "rb") as file:
    data = pickle.load(file)
    model = data['best_model']
    scaler = data['scaler']
    poly_transformer = data['poly_transformer']


def str_column_to_float(str_value):
    '''
     функция переводит str в float
     (удаляет единицы измерения в 'mileage', 'engine', 'max_power')
    '''
    if isinstance(str_value, str):
        try:
            return float(str_value.split()[0])
        except:
            return None
    return str_value


def preprocess_data(df):
    '''
    переводит 'mileage', 'engine', 'max_power' из строки в float
    если есть пропуски в 'mileage', 'engine', 'max_power', 'seats', заменяет их на медиану из трейна
    'engine' и 'seats' переводит в int
    удаляет 'torque', 'name', 'selling_price'
    оставляет только вещественные признаки
    преобразовывает данные согласно согласно сохраненной модели
    '''
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(str_column_to_float)

    medians = {
        'mileage': 19.37,
        'engine': 1248.0,
        'max_power': 81.86,
        'seats': 5.0
    }

    for col, median in medians.items():
        df[col] = df[col].fillna(median)

    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    df = df.drop(columns=['torque', 'name', 'selling_price'])
    df = df.select_dtypes('number')
    df_poly = poly_transformer.transform(df)
    df_scaled = scaler.transform(df_poly)
    return df_scaled


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_pr = preprocess_data(pd.DataFrame([item.model_dump()]))
    return model.predict(df_pr)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame([item.model_dump() for item in items])
    df_pr = preprocess_data(df)
    return model.predict(df_pr)