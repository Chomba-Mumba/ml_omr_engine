# use aws deep learning container
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-gpu-py310-cu125-ubuntu22.04-ec2

COPY . .

#install non tensorflow dependencies
RUN pip install -r requirements.txt

CMD ["python", "train.py"]