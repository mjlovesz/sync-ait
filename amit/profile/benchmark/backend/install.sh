#!/bin/bash

declare -i ret_ok=0
declare -i ret_run_failed=1
PYTHON_COMMAND="python3"

check_python_package_is_install()
{
    local PYTHON_COMMAND=$1
    ${PYTHON_COMMAND} -c "import $2" >> /dev/null 2>&1
    ret=$?
    if [ $ret != 0 ]; then
        echo "python package:$2 not install"
        return 1
    fi
    return 0
}

check_env_valid()
{
    check_command_exist atc || { echo "atc cmd not valid"; return $ret_run_failed; }
    check_python_package_is_install ${PYTHON_COMMAND} "aclruntime" \
    || { echo "aclruntime package not install"; return $ret_run_failed;}
}

main()
{
    check_env_valid
    res='echo $?'
    if [ $res =  $ret_run_failed ]; then
        pip3 whell ./ -v
        pip3 install ./aclruntime-*.whl
    fi
}

main "$@"
exit $?
}