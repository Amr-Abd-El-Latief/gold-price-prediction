# Steps to run the cluster
- install docker on master and worker nodes
- edit both .env files (Cluster/.env, Cluster/code/.env) to have IP of the master node
- Put docker-compose-worker.yml and Cluster/.env in the worker node
- run `docker compose -f docker-compose-master.yml up` in master node
- run `docker compose -f docker-compose-worker.yml up` in worker node
- open a new terminal in master node and run `docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/code/Linear-regression-gold.py` to run linear regression on both nodes
- Open http://localhost:8080, http://localhost:8081 in browser on master node
- Open http://localhost:8082 in browser in worker node
- Reload while working to see allocated resources & watch logs to see output of execution
