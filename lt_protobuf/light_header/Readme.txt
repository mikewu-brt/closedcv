
To compile the protobuf python headers, use the following:

protoc -I=<src_dir>/light_header --python_out=<closedcv_dir>/protobuf <src_dir>/light_header/*.proto

where:
<src_dir> is the path to the "proto_buf" repository
<closedcv_dir> is the path to the closedcv project


Examples:

protoc -I=./light_header --python_out=../closedcv/protobuf light_header/*.proto

protoc -I=~/Documents/PycharmProjects/proto_buf/light_header --python_out=~/Documents/PycharmProjects/closedcv/protobuf \
                    ~/Documents/PycharmProjects/proto_buf/light_header/*.proto

