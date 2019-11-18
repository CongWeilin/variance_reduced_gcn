python train_exps.py --dataset $1 --n_layers $2 --sample_method full 
python train_exps.py --dataset $1 --n_layers $2 --sample_method graphsage --samp_num 5 
python train_exps.py --dataset $1 --n_layers $2 --sample_method fastgcn --samp_num 512 
python train_exps.py --dataset $1 --n_layers $2 --sample_method ladies --samp_num 512 

python train_exps.py --dataset $1 --n_layers $2 --sample_method graphsage --samp_num 5 --variance_reduction True
python train_exps.py --dataset $1 --n_layers $2 --sample_method fastgcn --samp_num 512 --variance_reduction True
python train_exps.py --dataset $1 --n_layers $2 --sample_method ladies --samp_num 512 --variance_reduction True

python draw_train_plot.py $1 $2

rm *.pkl
rm *.pt