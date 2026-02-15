#!/bin/bash

stty -F /dev/ttyACM0 460800 raw -echo
cat /dev/ttyACM0

