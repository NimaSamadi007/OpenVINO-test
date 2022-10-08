#!/bin/bash 

file_addr=/tmp/cpu-flags.txt
if [ -e $file_addr ]
then
	rm $file_addr
fi
lscpu | grep "Model name:" >> $file_addr
lscpu | grep "Flags:" >> $file_addr
sed -i -E "s/\s{2,}/\n/g" $file_addr
sed -i -E "4s/ /\n/g" $file_addr
cat $file_addr
