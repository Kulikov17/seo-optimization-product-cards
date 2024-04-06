#!/bin/bash

kubectl apply -f ./k8s/namespace.yaml
kubectl apply -f ./k8s/secrets.yaml

kubectl apply -f ./k8s/postgres-config-map.yaml
kubectl apply -f ./k8s/postgres-deployment.yaml
kubectl apply -f ./k8s/postgres-service.yaml

kubectl apply -f ./k8s/backend-deployment.yaml
kubectl apply -f ./k8s/backend-service.yaml  

kubectl apply -f ./k8s/rabbitmq-deployment.yaml
kubectl apply -f ./k8s/rabbitmq-service.yaml  

kubectl apply -f ./k8s/celery-worker-deployment.yaml
kubectl apply -f ./k8s/celery-worker-service.yaml

kubectl apply -f ./k8s/flower-deployment.yaml
kubectl apply -f ./k8s/flower-service.yaml

kubectl apply -f ./k8s/nginx-config-map.yaml
kubectl apply -f ./k8s/nginx-deployment.yaml
kubectl apply -f ./k8s/nginx-service.yaml

kubectl apply -f ./k8s/telegram-deployment.yaml
kubectl apply -f ./k8s/telegram-service.yaml
