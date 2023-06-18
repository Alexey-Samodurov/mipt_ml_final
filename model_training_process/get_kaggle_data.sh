#!/usr/bin/env bash

kaggle datasets download -d arnabchaki/tripadvisor-reviews-2023
unzip -qq tripadvisor-reviews-2023.zip -d ../data/
rm tripadvisor-reviews-2023.zip

