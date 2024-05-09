FROM python:3.9

# 
WORKDIR /app

# 
COPY ./requirements.txt /app/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 
COPY . /app

# download the model weights in the image
RUN python /app/app/model/model.py

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]