FROM python:3.9-slim@sha256:980b778550c0d938574f1b556362b27601ea5c620130a572feb63ac1df03eda5 

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

ENV PORT 80

RUN pip install -r requirements.txt --cache-dir __pycache__ 

# As an example here we're running the web service with one worker on uvicorn.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1

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