#!/bin/bash

make html
git checkout gh-pages
cp -r build/html/* ..
git add ..
git commit -m "Update docs"
git push
git checkout master
