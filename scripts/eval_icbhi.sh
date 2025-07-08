######## EVAL ICBHI OPERA ########
echo extracting feature from $pretrain_model for downstream Task7;
python -u src/benchmark/processing/icbhi_processing.py  --pretrain operaCE --dim 1280

echo linear evaluation of $pretrain_model on downstream Task7;
python src/benchmark/linear_eval.py --task icbhidisease  --pretrain operaCE --dim 1280
################################
