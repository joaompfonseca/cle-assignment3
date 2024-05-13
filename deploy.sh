#!/bin/bash

NODE="cle13@banana.ua.pt"
FOLDER="cle-assignment3"

echo "-- Gathering files locally"
rm -rf dist
mkdir -p dist/$FOLDER
cp -r prog* dist/$FOLDER

echo "-- Compressing files locally"
cd dist
zip -rq $FOLDER.zip $FOLDER
cd ..

echo "-- Transferring files to BANANA"
sshpass -f password ssh "$NODE" "rm -rf ~/$FOLDER"
sshpass -f password scp -r dist/$FOLDER.zip "$NODE":~

echo "-- Decompressing files in BANANA"
sshpass -f password ssh "$NODE" "unzip -q ~/$FOLDER.zip -d ~"
sshpass -f password ssh "$NODE" "rm -rf ~/$FOLDER.zip"