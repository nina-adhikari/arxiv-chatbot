from typing import Union
from fastapi import FastAPI
import query

app = FastAPI()


@app.get("/")
async def root():
    return "Connected"
    #return query.connect("What is known about Euler characteristics of Hilbert schemes?")


# @app.get("/query/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return "Connected"
#     #return query.connect(q)


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}