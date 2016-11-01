#!/bin/bash

rm -rf ../Analises/*

for ((i = 1; i < 26; i++));
do
	mkdir ../Analises/${i}
	./bin/srpv ../config/config${i}.json ../Images/${i}.mp4 ../Analises/${i}/ 0 2
done
