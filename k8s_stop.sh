kubectl delete --all deployments -n seo 
kubectl delete --all services -n seo 

kubectl delete -f ./k8s/postgres-config-map.yaml
kubectl delete -f ./k8s/nginx-config-map.yaml
