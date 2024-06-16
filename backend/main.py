from fastapi import FastAPI
# from redis import Redis
# import msgpack

import query

# redis_url = "redis://localhost:6379"
# redi = Redis.from_url(redis_url)

app = FastAPI()
USER_COUNTER = 0
simple_dict = {}

# @app("startup")
# async def startup():
#     app.state.redis = await Redis.from_url(redis_url)

# @app("shutdown")
# async def shutdown():
#     await app.state.redis.close()

# async def get_redis():
#     return app.state.redis

@app.get("/")
async def root():
    return "Connected"

# @app.get("/query/{q}")
# async def read_item(q: str = None):
#     #return "Connected"
#     #result = await query.connect(q)
#     result = query.connect(q)
#     return result
#     # async for chunk in result:
#     #     yield chunk

@app.post("/query/")
async def chat(user_id: int, message: str):
    global USER_COUNTER

    if user_id != 0:
        agent = simple_dict[user_id]
        # retrieved_packed = redi.get(user_id)
        # state_dict = msgpack.unpackb(retrieved_packed)  # Unpack byte stream to dictionary
        # agent = query.ConversationAgentWrapper(**state_dict)  # Create new object from dictionary
    else:
        USER_COUNTER += 1
        user_id = USER_COUNTER
        agent = query.ConversationAgentWrapper()
        simple_dict[user_id] = agent
        # state_packed = msgpack.packb(agent.__dict__)  # Pack object dictionary
        # redi.set(user_id, state_packed)

    response = query.connect(agent, message)
    response['user_id'] = user_id
    return response