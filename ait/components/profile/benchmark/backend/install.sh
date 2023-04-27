#!/bin/bash

declare -i ret_ok=0
declare -i ret_run_failed=1

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
    check_python_package_is_install ${PYTHON_COMMAND} "aclruntime" \
    || { echo "aclruntime package not install"; return $ret_run_failed;}
}

main()
{
      while [ -n "$1" ]
do
  case "$1" in
    -p|--python_command)
        PYTHON_COMMAND=$2
        shift
        ;;
    *)
        echo "$1 is not an option, please use --help"
        exit 1
        ;;
  esac
  shift
done

    [ "$PYTHON_COMMAND" != "" ] || { PYTHON_COMMAND="python3.7";echo "set default pythoncmd:$PYTHON_COMMAND"; }

    check_env_valid
    res=`echo $?`
    if [ $res =  $ret_run_failed ]; then
        pip3 whell ./ -v
        pip3 install ./aclruntime-*.whl --force-reinstall
    fi
}

main "$@"
exit $?