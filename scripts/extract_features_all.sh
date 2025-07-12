
## runing baselines
sh scripts/extract_features.sh opensmile  >> cks/logs/result_opensmile.log
sh scripts/extract_features.sh vggish  >> cks/logs/result_vggish.log
sh scripts/extract_features.sh audiomae  >> cks/logs/result_audiomae.log
sh scripts/extract_features.sh clap  >> cks/logs/result_clap.log


## runing opera-X
sh scripts/extract_features.sh operaCT 768  >> cks/logs/result_operaCT.log
sh scripts/extract_features.sh operaCE 1280  >> cks/logs/result_operaCE.log
sh scripts/extract_features.sh operaGT 384 >> cks/logs/result_operaGT.log
