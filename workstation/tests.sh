#!/bin/bash

k=$(sed -n -e "1 p" Pf_params.txt)
echo "-Pf ${k}"