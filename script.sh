#!/bin/bash

for i in 1 
do
    sh scripts/SeparableTCN/PSM.sh
    sh scripts/SeparableTCN/SMD.sh
    sh scripts/SeparableTCN/SWaT.sh
done
