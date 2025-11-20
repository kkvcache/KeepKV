jsonl=../output/xsum-llama-full.jsonl 
task=xsum                   
model_arch=llama            
output_name=xsum-llama-full 

python scripts/offline_eval/import_results.py together ${jsonl} --cache-dir prod_env/cache 

helm-run --conf src/helm/benchmark/presentation/${task}/run_specs_${model_arch}.conf --local --max-eval-instances 1000 --num-train-trials=1 --suite ${output_name} -n 1

helm-summarize --suite ${output_name}