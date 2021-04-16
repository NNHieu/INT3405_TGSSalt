#!/bin/bash

newgrp docker
docker container start c3b
docker logs c3b | tee log.txt
