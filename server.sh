pip install pymatgen
pip install ase

nohup python -u main.py > log 2>&1 &

nohup python -u train.py > log 2>&1 &

nohup python -u serctrain.py > log 2>&1 &
nohup python -u sercvalid.py > log 2>&1 &


tail -n 50 -f log

ls -l | wc -l

tar -czvf main.tar.gz ./

ps -ef | grep python | cut -c 9-15| xargs kill -s 9


pip install sklearn
pip install iterative-stratification==0.1.6


find -name "*.x" | wc -l

tensorflow-gpu 2.6.0 requires numpy~=1.19.2, but you have numpy 1.23.4 which is incompatible.
tensorflow-gpu 2.6.0 requires typing-extensions~=3.7.4, but you have typing-extensions 4.6.3 which is incompatible.
