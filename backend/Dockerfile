# 
# FROM python:3.9

# # 
# WORKDIR /code

# # 
# COPY ./requirements.txt /code/requirements.txt

# # 
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# # 
# COPY ./app /code/app

# 
#CMD ["fastapi", "run", "app/main.py", "--port", "80"]

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# CMD ["gunicorn", \
#     "-k", "uvicorn.workers.UvicornWorker", \
#     "--bind", "0.0.0.0:80", \
#     "app.main:app" \
#     ]

#CMD gunicorn -k uvicorn.workers.UvicornWorker --bind "0.0.0.0:8080" --log-level debug app.main:app

FROM python:3.9

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

ENV PORT 1234

RUN pip install --no-cache-dir -r requirements.txt

# As an example here we're running the web service with one worker on uvicorn.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1