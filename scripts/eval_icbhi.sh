eval_type=$1

if [ "$eval_type" == "noisy" ]; then
    model_name="operaCT"
    dim=768
    python src/benchmark/linear_noisy_eval.py --task icbhidisease --pretrain $model_name --dim $dim

    model_name="operaCE"
    dim=1280
    python src/benchmark/linear_noisy_eval.py --task icbhidisease --pretrain $model_name --dim $dim

    model_name="operaGT"
    dim=384
    python src/benchmark/linear_noisy_eval.py --task icbhidisease --pretrain $model_name --dim $dim

    list="opensmile vggish clap audiomae"
    for i in $list;
    do
    python src/benchmark/linear_noisy_eval.py --task icbhidisease --pretrain $i
    done

# else
#     model_name="operaCT"
#     dim=768
#     python src/benchmark/processing/icbhi_processing.py --pretrain $model_name --dim $dim
#     python src/benchmark/linear_eval.py --task icbhidisease --pretrain $model_name --dim $dim

#     model_name="operaCE"
#     dim=1280
#     python src/benchmark/processing/icbhi_processing.py --pretrain $model_name --dim $dim
#     python src/benchmark/linear_eval.py --task icbhidisease --pretrain $model_name --dim $dim

#     model_name="operaGT"
#     dim=384
#     python src/benchmark/processing/icbhi_processing.py --pretrain $model_name --dim $dim
#     python src/benchmark/linear_eval.py --task icbhidisease --pretrain $model_name --dim $dim

#     list="opensmile vggish clap audiomae"
#     for i in $list;
#     do
#     python src/benchmark/processing/icbhi_processing.py --pretrain $i 
#     python src/benchmark/linear_eval.py --task icbhidisease --pretrain $i
#     done
fi