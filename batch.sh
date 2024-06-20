#!/bin/bash

set -e

universities=('UFPR04' 'UFPR05' 'PUC')
#models=('simple' 'threecvnnencoder')
models=('threecvnnencoder')
#date="$(date --iso-8601=minutes)"
date="managed_test"
percent_train="0.1"
percent_test="0.125"
percent_validation="0.025"
results_path="batch_results/${date}"
#dataset_json="${results_path}/dataset_${date}_${university}.json"
epochs="60"

printf "Universities:"
for university in ${universities[@]}
do
  printf " $university"
done
printf "\n"

printf "Models:"
for model in ${models[@]}
do
  printf " $model"
done
printf "\n"

#printf "Jsons:"
#for university in ${universities[@]}
#do
#  printf " ${dataset_json}"
#done
#printf "\n"

echo "Date: ${date}"
echo "Train percentage: ${percent_train}"
echo "Test percentage: ${percent_test}"
echo "Validation percentage: ${percent_validation}"
echo "Epochs: ${epochs}"

#read -p "Press enter to continue..."


#mkdir "${results_path}"

#for university in ${universities[@]}
#do
#  python3 utils.py -u "${university}" -g True -tr "${percent_train}" -tt "${percent_test}" \
#             -vf "${percent_validation}" -o "${results_path}/dataset_${university}.json" \
#             -ov False -dp ./
#done
#
##read -p "Press enter to continues..."
#
#for i in {1..3}
#do
#  for model in ${models[@]}
#  do
#    for university in ${universities[@]}
#    do
#      mkdir --parents "${results_path}/${model}/${university}_${i}"
#      python3 main.py --model-path "${results_path}/${model}/${university}_${i}" \
#                      --log True \
#                      --log-path "${results_path}/${model}/${university}_${i}/log.txt" \
#                      --model "${model}" \
#                      --random True \
#                      --dataset-json "${results_path}/dataset_${university}.json" \
#                      --epochs "${epochs}"
#    done
#  done
#done


for j in {1..3}
do
  for i in {1..3}
  do
    for model in ${models[@]}
    do
      for university in ${universities[@]}
      do
        mkdir --parents "${results_path}/threecvnnclassifier/${university}_${j}_${i}"
        python3 main.py --model-path "${results_path}/threecvnnclassifier/${university}_${j}_${i}" \
                        --log True \
                        --log-path "${results_path}/threecvnnclassifier/${university}_${j}_${i}/log.txt" \
                        --model threecvnnclassifier \
                        --random True \
                        --pretrained-weights "${results_path}/${model}/${university}_${i}/best.weights.h5" \
                        --dataset-json "${results_path}/dataset_${university}.json" \
                        --epochs "${epochs}"
      done
    done
  done
done
