# 🌟 **Unsupervised Data Augmentation for Text Classification with BERT**

## 📖 **Overview**
This project explores **Unsupervised Data Augmentation (UDA)** techniques to enhance text classification performance using **BERT**. UDA leverages advanced data augmentation methods, such as **back-translation** and **TF-IDF based word replacement**, to utilize large amounts of unlabeled data. This allows for improved model generalization, reduced reliance on labeled data, and enhanced classification accuracy.

The project demonstrates UDA’s effectiveness across several well-known datasets and compares its performance against traditional supervised learning methods.

---

## ✨ **Features**
- 🚀 **Implementation of UDA** for semi-supervised text classification.
- 🤖 Utilization of **BERT** pre-trained models (BERT-Base and BERT-Large).
- 🔄 **Advanced augmentation techniques**:
  - 🔁 **Back-translation**: Generates paraphrased text for data diversity.
  - ✍️ **TF-IDF word replacement**: Replaces low-information words while preserving critical keywords.
- 📊 **Evaluated on diverse datasets**:
  - 🎥 **IMDb**: Binary sentiment classification.
  - 📝 **Yelp-2 and Yelp-5**: Sentiment analysis.
  - 🛒 **Amazon-2 and Amazon-5**: Product review classification.
  - 📚 **DBpedia**: Topic classification.
- ⚡ **Error rate reductions** and accuracy improvements across datasets.
- ⚙️ **Flexible Hyperparameters**: Control augmentation diversity, unsupervised loss weight, batch size, etc.
- 🔋 **GPU & TPU Support**: Easily switch between local GPU training or large-scale TPU v3-32 Pod training.
- 💡**High Accuracy**: Achieve around 90% accuracy on baseline GPU and 95%+ with TPU + UDA.


---

## 📚 **Datasets**
The project uses multiple text classification datasets:
1. 🎥 **IMDb**: Binary sentiment classification for movie reviews.
2. 📝 **Yelp-2 and Yelp-5**: Binary and multi-class sentiment analysis of customer reviews.
3. 🛒 **Amazon-2 and Amazon-5**: Binary and multi-class product review classification.
4. 📚 **DBpedia**: Topic classification from structured data.

---



## 🎉 Requirements

The code is tested on **🐍 Python 3.8 or above.** and **Tensorflow 1.13**.  

After installing Tensorflow, run the following commands to install dependencies:

```bash
pip install --user absl-py
pip install fire tqdm tensorboardX pandas numpy
pip install tensorflow  # or tensorflow-gpu if you have a compatible GPU
pip install torch torchvision  # This installs PyTorch
```
## ⚙️ Instructions

To run **UDA** with **BERT** base on a GPU with **11 GB memory**, use the following commands:

```bash
# Set a larger max_seq_length if your GPU has more than 11GB memory
MAX_SEQ_LENGTH=128

# Download data and pretrained BERT checkpoints
bash scripts/download.sh

# Preprocessing
bash scripts/prepro.sh --max_seq_length=${MAX_SEQ_LENGTH}

# Baseline accuracy: around 68%
bash scripts/run_base.sh --max_seq_length=${MAX_SEQ_LENGTH}

# UDA accuracy: around 90%
# Set a larger train_batch_size to achieve better performance if your GPU has a larger memory.
bash scripts/run_base_uda.sh --train_batch_size=8 --max_seq_length=${MAX_SEQ_LENGTH}
```


## ☁️ Run on Cloud TPU v3-32 Pod

For the best performance,use a `max_seq_length` of **512** and initialize with **BERT large** ,finetuned on in-domain unsupervised data.

If you have access to **Google Cloud TPU v3-32 Pod**, run the following commands:

```bash
MAX_SEQ_LENGTH=512

# Download data and pretrained BERT checkpoints
bash scripts/download.sh

# Preprocessing
bash scripts/prepro.sh --max_seq_length=${MAX_SEQ_LENGTH}

# UDA accuracy: 95.3% - 95.9%
bash train_large_ft_uda_tpu.sh
```

## 🔀 Run Back Translation Data Augmentation on the Dataset

To perform back translation for data augmentation, install the following dependencies:


```bash
pip install --user nltk
python -c "import nltk; nltk.download('punkt')"
pip install --user tensor2tensor==1.13.4
```
How it works:

1. Splits paragraphs into sentences.
2. Translates English sentences to French.
3. Translates them back into English.
4. Composes the paraphrased sentences back into paragraphs.
5. Go to the back_translate directory and run:

## Run Back Translation
Navigate to the back_translate directory and execute the following commands:

```bash
bash download.sh
bash run.sh
```

## ⚙️ Guidelines for Hyperparameters

**sampling_temp**: Controls the diversity and quality of paraphrases. Increasing `sampling_temp` increases diversity but may reduce quality.  
- Recommended values: `sampling_temp = 0.7, 0.8, or 0.9`.  
- For tasks with high quality requirements, `sampling_temp = 0.9` or `0.8` should lead to improved performance.  

**unsup_coeff**: Controls the weight of unsupervised loss.
- Set to **1** for balanced supervised and unsupervised losses.

**Learning Rate**: Use a lower learning rate than pure supervised learning because there are two loss terms (labeled + unlabeled data).
- If you have a **small dataset**, tweak `uda_softmax_temp` and `uda_confidence_thresh`.

**uda_softmax_temp** and **uda_confidence_thresh**: Controls the diversity and quality of paraphrases. Increasing `uda_softmax_temp` increases diversity but may reduce quality.

To perform back translation on a large file, adjust `replicas` and `worker_id` in `run.sh`. For example, `replicas=3` divides the data into three parts, and each `run.sh` processes one part based on the `worker_id`.

**Running back translation on a large file**
To Process large datsets adjust the `replicas` and `worker_id` in `run.sh`.:
For example, `replicas=3` divides the data into three parts, and each `run.sh` processes one part based on the `worker_id`.
---

## 📌 General Guidelines for Setting Hyperparameters
1. **Batch Size**: If your GPU has a larger memory, increase `train_batch_size` to improve performance.
2. **Sequence Length**: Use larger `max_seq_length` Especially for tasks requiring deep contextual understanding.
3. **Effective augmentation** : Standard augmentations from supervised learning often work well with UDA when combined with the unsupervised objective.


---

## 🙏 Acknowledgement

This project builds on and integrates components from **BERT** and **RandAugment**. We appreciate their contributions to the community.

**Thank you!**

## 📚 Citation

```bibtex
@article{xie2019unsupervised,
  title={Unsupervised Data Augmentation for Consistency Training},
  author={Xie, Qizhe and Dai, Zihang and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
  journal={arXiv preprint arXiv:1904.12848},
  year={2019}
}
