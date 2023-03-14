epoch=50
dpi=300
windowsize=20
batch_size=10000
save_root=save/23.03.14/absolute_err/
gpu=0

python main.py -v $windowsize -g $gpu -e $epoch -s $save_root'wadi.new.lstmcvae' --which_model lstmcvae -S --dataset wadi.new -GNDTA -d $dpi -b $batch_size
python main.py -v $windowsize -g $gpu -e $epoch -s $save_root'wadi.new.cvae'     --which_model cvae     -S --dataset wadi.new -GNDTA -d $dpi -b $batch_size
python main.py                -g $gpu -e $epoch -s $save_root'wadi.new.vae'      --which_model vae      -S --dataset wadi.new -GNDTA -d $dpi -b $batch_size
python main.py -v $windowsize -g $gpu -e $epoch -s $save_root'wadi.old.lstmcvae' --which_model lstmcvae -S --dataset wadi.old -GNDTA -d $dpi -b $batch_size
python main.py -v $windowsize -g $gpu -e $epoch -s $save_root'wadi.old.cvae'     --which_model cvae     -S --dataset wadi.old -GNDTA -d $dpi -b $batch_size
python main.py                -g $gpu -e $epoch -s $save_root'wadi.old.vae'      --which_model vae      -S --dataset wadi.old -GNDTA -d $dpi -b $batch_size
python main.py -v $windowsize -g $gpu -e $epoch -s $save_root'swat.lstmcvae'     --which_model lstmcvae -S --dataset swat     -GNDTA -d $dpi -b $batch_size
python main.py -v $windowsize -g $gpu -e $epoch -s $save_root'swat.cvae'         --which_model cvae     -S --dataset swat     -GNDTA -d $dpi -b $batch_size
python main.py                -g $gpu -e $epoch -s $save_root'swat.vae'          --which_model vae      -S --dataset swat     -GNDTA -d $dpi -b $batch_size


