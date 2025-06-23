# use aws deep learning container
FROM public.ecr.aws/deeplearning-containers/tensorflow-training:2.18.0-gpu-py310-cu125-ubuntu22.04-ec2

COPY . .

#install non tensorflow dependencies
RUN pip install -r requirements.txt

CMD ["python", "train.py"]