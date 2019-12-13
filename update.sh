#!/usr/bin/env bash

python3 -m grpc_tools.protoc -I../../protos --python_out=. --grpc_python_out=. proto/service.proto
