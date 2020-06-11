#!/bin/bash

docker exec -it `docker ps |grep adev-proj:latest|cut -d ' ' -f 1`  bash
