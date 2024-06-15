from typing import Union
from fastapi import FastAPI
import query

app = FastAPI()


@app.get("/")
async def root():
    return "Connected"
    #return query.connect("What is known about Euler characteristics of Hilbert schemes?")


@app.get("/query/{q}")
async def read_item(q: str = None):
    #return "Connected"
    #result = await query.connect(q)
    result = query.connect(q)
    return result
    # async for chunk in result:
    #     yield chunk


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}