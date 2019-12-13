# python train.py --dataset ppi --sample_method ladies 
# python train.py --dataset ppi-large --sample_method ladies
# python train.py --dataset flickr --sample_method ladies
# python train.py --dataset reddit --sample_method ladies --cuda -1
# python train.py --dataset yelp --sample_method ladies --cuda -1

python train.py --dataset ppi --sample_method fastgcn 
python train.py --dataset ppi-large --sample_method fastgcn
python train.py --dataset flickr --sample_method fastgcn
# python train.py --dataset reddit --sample_method fastgcn --cuda -1
# python train.py --dataset yelp --sample_method fastgcn --cuda -1

python train.py --dataset ppi --sample_method graphsage 
python train.py --dataset ppi-large --sample_method graphsage
python train.py --dataset flickr --sample_method graphsage
# python train.py --dataset reddit --sample_method graphsage --cuda -1
# python train.py --dataset yelp --sample_method graphsage --cuda -1

python train.py --dataset ppi --sample_method graphsaint
python train.py --dataset ppi-large --sample_method graphsaint
python train.py --dataset flickr --sample_method graphsaint
# python train.py --dataset reddit --sample_method graphsaint --cuda -1
# python train.py --dataset yelp --sample_method graphsaint --cuda -1