#!/bin/bash

docker stop `docker ps |grep adev-proj:latest|cut -d ' ' -f 1`
