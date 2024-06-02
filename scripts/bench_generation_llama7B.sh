cd $1/experiments
python replace_util.py --root $1 --function backup
python bench_sparse_inference_llama.py --B 1 --bm 1 --gen 257 --method 2 --root_path $1 --weights_dir $2 --results_dir $3
python bench_sparse_inference_llama.py --B 1 --bm 1 --gen 513 --method 2 --root_path $1 --weights_dir $2 --results_dir $3
python bench_sparse_inference_llama.py --B 1 --bm 1 --gen 1025 --method 2 --root_path $1 --weights_dir $2 --results_dir $3
python bench_sparse_inference_llama.py --B 1 --bm 1 --gen 2049 --method 2 --root_path $1 --weights_dir $2 --results_dir $3
python bench_dense_inference_llama.py --B 1 --bm 1 --gen 257
python bench_dense_inference_llama.py --B 1 --bm 1 --gen 513
python bench_dense_inference_llama.py --B 1 --bm 1 --gen 1025
python bench_dense_inference_llama.py --B 1 --bm 1 --gen 2049
python replace_util.py --root $1 --function restore