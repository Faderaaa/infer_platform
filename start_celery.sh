nohup celery -A app.celery flower --address=127.0.0.1 --port=5566  > ./log/flower.log 2>&1 &
sleep 2
nohup celery -A app.celery  worker --loglevel=INFO  > ./log/celery.log 2>&1 &
