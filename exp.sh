#!/bin/bash
dpi=300
#windowsize=20
batch_size=10000
save_root=save/23.03.16/window/
gpu=3
epoch=50

#python main.py -v $windowsize -g $gpu -e $epoch -s $save_root'swat.lstmcvae'     --which_model lstmcvae  --dataset swat     -GNDTASO -d $dpi -b $batch_size
#python main.py -v $windowsize -g $gpu -e $epoch -s $save_root'swat.lstmcvae'     --which_model lstmcvae  --dataset swat     -GNDTAS  -d $dpi -b $batch_size
#exit

for i in {1..10}
do
  echo $i
  windowsize=`expr ${i} \* 10`
  python main.py -v $windowsize -g $gpu -e $epoch -s $save_root$windowsize'/swat.lstmcvae'     --which_model lstmcvae --dataset swat     -GNDTASO -d $dpi -b $batch_size
  python main.py -v $windowsize -g $gpu -e $epoch -s $save_root$windowsize'/swat.cvae'         --which_model cvae     --dataset swat     -GNDTASO -d $dpi -b $batch_size
#  python main.py                -g $gpu -e $epoch -s $save_root$windowsize'/swat.vae'          --which_model vae      --dataset swat     -GNDTASO -d $dpi -b $batch_size
  python main.py -v $windowsize -g $gpu -e $epoch -s $save_root$windowsize'/wadi.new.lstmcvae' --which_model lstmcvae --dataset wadi.new -GNDTASO -d $dpi -b $batch_size
  python main.py -v $windowsize -g $gpu -e $epoch -s $save_root$windowsize'/wadi.new.cvae'     --which_model cvae     --dataset wadi.new -GNDTASO -d $dpi -b $batch_size
#  python main.py                -g $gpu -e $epoch -s $save_root$windowsize'/wadi.new.vae'      --which_model vae      --dataset wadi.new -GNDTASO -d $dpi -b $batch_size
  python main.py -v $windowsize -g $gpu -e $epoch -s $save_root$windowsize'/wadi.old.lstmcvae' --which_model lstmcvae --dataset wadi.old -GNDTASO -d $dpi -b $batch_size
  python main.py -v $windowsize -g $gpu -e $epoch -s $save_root$windowsize'/wadi.old.cvae'     --which_model cvae     --dataset wadi.old -GNDTASO -d $dpi -b $batch_size
#  python main.py                -g $gpu -e $epoch -s $save_root$windowsize'/wadi.old.vae'      --which_model vae      --dataset wadi.old -GNDTASO -d $dpi -b $batch_size
done
