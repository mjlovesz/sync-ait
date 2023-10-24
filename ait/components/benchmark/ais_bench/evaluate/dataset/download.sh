#!/usr/bin/env bash

declare -i ret_ok=0
declare -i ret_error=1

dir=$(pwd)

function download()
{
    dataset=$1
    if [ -f "$dir/$dataset.tar.gz" ]; then
        rm "$dir/$dataset.tar.gz"
        echo "Removed existing file: $dir/$dataset.tar.gz"
    fi

    echo "dataset: $dataset now downloading"
    url="https://llm-dataset.obs.myhuaweicloud.com/$dataset.tar.gz"
    wget --no-check-certificate --tries=3 "$url" >/dev/null 2>&1

    ret=$?
    if [ $ret -ne 0 ]; then
        echo "wget failed to download file from URL: $url"
        return $ret
    else
        echo "dataset: $dataset downloaded successfully"
    fi
    return $ret_ok
}

function unzip()
{
    dataset=$1
    if [ -d $dir/$dataset ]; then
        rm -r $dir/$dataset
        echo "Removed existing directory: $dir/$dataset"
    fi

    tar -xzvf "$dir/$dataset.tar.gz" > /dev/null 2>&1

    ret=$?
    if [ $ret -ne 0 ]; then
        echo "tar failed to unzip $dir/$dataset.tar.gz"
        return $ret
    else
        echo "dataset: $dataset unzipped successfully"
    fi

    rm "$dir/$dataset.tar.gz"

    return $ret_ok
}

function main()
{
    if [ "$1" != "ceval" -a "$1" != "mmlu" -a "$1" != "gsm8k" ];then
        echo "Invalid dataset: [$1] does not match [ceval mmlu gsm8k]"
        return $ret_error
    fi
    dataset=$1

    download "$dataset"
    ret_download=$?

    if [ $ret_download -ne 0 ]; then
        return $ret_error
    fi

    unzip "$dataset"
    ret_unzip=$?

    if [ $ret_unzip -ne 0 ]; then
        return $ret_error
    fi

    return $ret_ok
}
main "$@"
exit $?