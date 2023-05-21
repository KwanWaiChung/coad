## CoAD: Automatic Diagnosis through Symptom and Disease Collaborative Generation
**Authors**:  Huimin Wang Wai-Chung Kwan, Kam-Fai Wong, Yefeng Zheng  

<!-- ## Citation:
If you find our paper and resources useful, please kindly cite our paper:
```bibtex
``` -->

### Environment Setup:
```yaml
conda env create --name coad --file environment.yml
```

### Dataset
We only download the dxy and muzhi dataset due to license issue.
```yaml
cd data
chmod u+x download_dataset.sh
./download_dataset.sh
```

### Training 
```yaml
# dataset: Either dxy, muzhi.
python train.py \
--dataset dxy \
--do_sym_augmentation \
--do_dis_augmentation \
--do_train
```

### Testing
```yaml
python train.py 
--dataset dxy \
--checkpoint $checkpoint_path \
--do_test
```