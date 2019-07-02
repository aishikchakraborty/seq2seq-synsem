#!/bin/bash
# mkdir -p data
cd data

# wget http://statmt.org/wmt13/training-parallel-europarl-v7.tgz
# tar -xzvf training-parallel-europarl-v7.tgz
# rm training-parallel-europarl-v7.tgz
#
# wget http://statmt.org/wmt14/dev.tgz
# tar -xzvf dev.tgz
# rm dev.tgz
sed -i -e 's/^/en /' training/europarl-v7.fr-en.en
sed -i -e 's/^/fr /' training/europarl-v7.fr-en.fr
sed -i -e 's/^/de /' training/europarl-v7.de-en.de
sed -i -e 's/^/en /' training/europarl-v7.de-en.en
cat training/europarl-v7.de-en.en training/europarl-v7.fr-en.en training/europarl-v7.de-en.de training/europarl-v7.fr-en.fr > training/train_all
cat training/europarl-v7.de-en.de training/europarl-v7.fr-en.en > training/train.src
cat training/europarl-v7.de-en.en training/europarl-v7.fr-en.fr > training/train.tgt

cd ..
