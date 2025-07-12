# to run the script, copy feature, scripts datasets and src directories
# script example for running: sh scripts/eval_all.sh OPERA-GT 1280  >> out/multiple150.txt

pretrain_model=$1
if [ $# -gt 1 ]; then
        dim=$2
        echo 'Feature dimension:' $dim
else
        dim=0
        echo 'Baseline: no need to specify dimension'
fi


# # task 3
# echo extracting feature from $pretrain_model for downstream Task3;
# python -u src/benchmark/processing/covid19sounds_processing.py --task 1  --pretrain $pretrain_model  --modality breath  --dim $dim

# # task 4
# echo extracting feature from $pretrain_model for downstream Task4;
# python -u src/benchmark/processing/covid19sounds_processing.py --task 1  --pretrain $pretrain_model  --modality cough  --dim $dim

# task 5
# echo extracting feature from $pretrain_model for downstream Task5;
# python -u src/benchmark/processing/coughvid_processing.py   --pretrain $pretrain_model  --label covid  --dim $dim

# # task 6
# echo extracting feature from $pretrain_model for downstream Task6;
# python -u src/benchmark/processing/coughvid_processing.py   --pretrain $pretrain_model  --label gender  --dim $dim

# task 7
echo extracting feature from $pretrain_model for downstream Task7;
python -u src/benchmark/processing/icbhi_processing.py  --pretrain $pretrain_model --dim $dim

# # task 8
# echo extracting feature from $pretrain_model for downstream Task8;
# python -u src/benchmark/processing/coswara_processing.py --pretrain $pretrain_model  --modality cough-shallow  --label smoker  --dim $dim

# # task 9
# echo extracting feature from $pretrain_model for downstream Task9;
# python -u src/benchmark/processing/coswara_processing.py --pretrain $pretrain_model --modality cough-shallow --label sex --dim $dim

# # task 10 
# echo extracting feature from $pretrain_model for downstream Task10;
# python -u src/benchmark/processing/kauh_processing.py --pretrain $pretrain_model --dim $dim

# task 11
echo extracting feature from $pretrain_model for downstream Task11;
python -u src/benchmark/processing/copd_processing.py --pretrain $pretrain_model --dim $dim

# # task 12
# echo extracting feature from $pretrain_model for downstream Task12;
# python -u src/benchmark/processing/ssbpr_processing.py --pretrain $pretrain_model --dim $dim

# # ### Task 13-18
# echo extracting feature from $pretrain_model for downstream Task13-18;
# python -u src/benchmark/processing/mmlung_processing.py --pretrain $pretrain_model --dim $dim


# # ### Task 19
# echo extracting feature from $pretrain_model for downstream Task19;
# python -u src/benchmark/processing/nosemic_processing.py --pretrain $pretrain_model --dim $dim
