FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt -t .

COPY app.py ./
COPY minimal_model.py ./
COPY models /opt/ml/models

CMD ["app.lambda_handler"]
