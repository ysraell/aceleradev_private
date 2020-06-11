#!/bin/bash

#./stop_CONTAINER.sh || echo 'No container with iamge adev-proj:latest running.'
#docker run -d -p 8080:8080 -p 8501:8501 -v $HOME:/work -t adev-proj:latest
#echo 'Wait 5 seconds for Jupyter Lab access...'
#sleep 5
docker exec -it `docker ps |grep adev-proj:latest|cut -d ' ' -f 1`  jupyter notebook list

