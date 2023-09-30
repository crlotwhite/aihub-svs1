# WAVEGLOW

레퍼런스로 제공된 베이스 모델입니다.
일부 코드를 수정하여 최신 pytorch 2.x 버전에서 작동되도록 수정한 버전입니다.


## 대충 스크립트
```bash
tar -xvf vvc.tar

git clone https://github.com/crlotwhite/aihub-svs1.git

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install tqdm==4.62.3 scipy==1.7.3 tensorboard==2.8.0 tensorboardX==2.4.1 librosa==0.8.1 pyworld==0.3.0

ls ~/vvc/SINGER_01/*.wav > train_file.txt

tail train_file.txt

python preprocess.py train_file.txt ~/vvc/SINGER_01/ -o preprocess


python train.py preprocess/config.json -g 0 -c checkpoint -b 32 -w 4 -t 4 --log log -lv

```