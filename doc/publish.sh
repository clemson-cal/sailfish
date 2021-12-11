#!/bin/bash

make html
cd ..
git checkout gh-pages
cp -r doc/build/html/* .
git add .
git commit -m "Update docs"
git push
git checkout master
cd doc
