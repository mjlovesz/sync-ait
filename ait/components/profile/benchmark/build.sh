# Copyright (c) 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
CURDIR=$(dirname $(readlink -f $0))
PLATFORM=`uname -i`
PACKET_NAME="ais_bench_$PLATFORM"
OUTPUT_PATH="$CURDIR/$PACKET_NAME/"

main()
{
    [ ! -d $OUTPUT_PATH ] || rm -rf $OUTPUT_PATH
    mkdir -p $OUTPUT_PATH

    rm -rf $CURDIR/*.whl
    cd $CURDIR
    rm -rf $CURDIR/backend/*.egg-info $CURDIR/backend/build
    which pip3.7 && { pip3.7 wheel -v $CURDIR/backend/ || echo "pip3.7 run failed"; }
    which pip3.8 && { pip3.8 wheel -v $CURDIR/backend/ || echo "pip3.8 run failed"; }
    which pip3.9 && { pip3.9 wheel -v $CURDIR/backend/ || echo "pip3.9 run failed"; }

    rm -rf $CURDIR/*.egg-info $CURDIR/build 
    which pip3.7 && { pip3.7 wheel -v $CURDIR/ || echo "pip3.7 run failed"; }
    which pip3.8 && { pip3.8 wheel -v $CURDIR/ || echo "pip3.8 run failed"; }
    which pip3.9 && { pip3.9 wheel -v $CURDIR/ || echo "pip3.9 run failed"; }

    cp $CURDIR/aclruntime*.whl -rf $OUTPUT_PATH/
    cp $CURDIR/ais_bench*.whl -rf $OUTPUT_PATH/
    
    cp $CURDIR/ais_bench -rf $OUTPUT_PATH/
    cp $CURDIR/requirements.txt $OUTPUT_PATH/
    cp $CURDIR/README.md $OUTPUT_PATH/
    cp $CURDIR/FAQ.md $OUTPUT_PATH/

    cd $CURDIR
    rm -rf $CURDIR/$PACKET_NAME.tar.gz
    tar -czf $CURDIR/$PACKET_NAME.tar.gz $PACKET_NAME

    rm -rf ${CURDIR}/output/*
    mkdir -p ${CURDIR}/output
    cp $CURDIR/$PACKET_NAME.tar.gz ${CURDIR}/output
}

main "$@"
exit $?
