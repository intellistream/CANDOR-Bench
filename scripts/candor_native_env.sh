#!/usr/bin/env bash

# Runtime library paths for native ANN bindings. Source this before `uv run`.

_candor_prepend_path() {
    local var_name="$1"
    local value="$2"
    [ -n "$value" ] || return 0
    [ -d "$value" ] || return 0

    eval "local current=\"\${$var_name:-}\""
    case ":$current:" in
        *":$value:"*) ;;
        *) export "$var_name=$value${current:+:$current}" ;;
    esac
}

_candor_prepend_preload() {
    local value="$1"
    [ -n "$value" ] || return 0
    [ -f "$value" ] || return 0

    case ":${LD_PRELOAD:-}:" in
        *":$value:"*) ;;
        *) export LD_PRELOAD="$value${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
}

if [ -d "/opt/intel/oneapi/mkl/latest" ]; then
    export MKLROOT="/opt/intel/oneapi/mkl/latest"
    _candor_prepend_path LD_LIBRARY_PATH "$MKLROOT/lib/intel64"
    _candor_prepend_path LIBRARY_PATH "$MKLROOT/lib/intel64"
    _candor_prepend_path CPATH "$MKLROOT/include"
    _candor_prepend_path CMAKE_PREFIX_PATH "$MKLROOT"
elif [ -d "/opt/intel/mkl" ]; then
    export MKLROOT="/opt/intel/mkl"
    _candor_prepend_path LD_LIBRARY_PATH "$MKLROOT/lib/intel64"
    _candor_prepend_path LIBRARY_PATH "$MKLROOT/lib/intel64"
    _candor_prepend_path CPATH "$MKLROOT/include"
    _candor_prepend_path CMAKE_PREFIX_PATH "$MKLROOT"
fi

for _candor_compiler_lib in \
    "/opt/intel/oneapi/compiler/2025.3/lib" \
    "/opt/intel/oneapi/compiler/latest/lib"
do
    if [ -d "$_candor_compiler_lib" ]; then
        _candor_prepend_path LD_LIBRARY_PATH "$_candor_compiler_lib"
        _candor_prepend_preload "$_candor_compiler_lib/libiomp5.so"
        break
    fi
done

_candor_prepend_preload "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

for _candor_torch_lib in "$PWD"/.venv/lib/python*/site-packages/torch/lib; do
    _candor_prepend_path LD_LIBRARY_PATH "$_candor_torch_lib"
done

unset _candor_compiler_lib
unset _candor_torch_lib
