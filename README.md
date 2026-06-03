# Building age estimation
A novel building age estimation model based on remote sensing data

This project includes the code for ten image-based models, Unet, Unetformer, ABCNet, Segformer, CMNeXt, TransUnet, CMTFNet, CMT, Deeplab, A2FPN, and three multi-modal-based models, namely CMX, Asymformer, MFNet for building age estimation. Additionally, it incorporates the data loaders for amsterdam, and a custom dataset named Hong Kong. The respective papers for these models and the download links for the datasets are provided below.


**datasets**

* Hong Kong Building age dataset [download link](https://huggingface.co/datasets/littlebeen/Buildingage/tree/main)

**models**

* Unet : [U-net: Convolutional networks for biomedical image segmentation]

* Unetformer : [UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery]

* ABCNet : [ABCNet: Attentive bilateral contextual network for efficient semantic segmentation of fine-resolution remotely sensed imagery]

* Segformer : [SegFormer: Simple and efficient design for semantic segmentation with transformers]

* CMNeXt : [Delivering arbitrary-modal semantic segmentation]

* TransUnet : [Transunet: Transformers make strong encoders for medical image segmentation]

* CMTFNet : [CMTFNet: CNN and multiscale transformer fusion network for remote-sensing image semantic segmentation]

* CMT : [Cmt: Convolutional neural networks meet vision transformers]

* Deeplab : [Encoder-decoder with atrous separable convolution for semantic image segmentation]

* A2FPN : [A2-FPN for semantic segmentation of fine-resolution remotely sensed images]

* CMX : [CMX: Cross-modal fusion for RGB-X semantic segmentation with transformers]

* Asymformer : [Asymformer: Asymmetrical cross-modal representation learning for mobile platform real-time rgb-d semantic segmentation]

* MFNet : [A unified framework with multimodal fine-tuning for remote sensing semantic segmentation]


 # Usage

**Train**

1. Change the mode to 'train' and choose the model you need in config.py
2. Prepare the dataset at 'my_dataset' (You can find the detailed guidance in my_dataset/readme.txt)
2. Put the pre-train model into 'weights', if the model need. (You can find the detailed guidance in weights/readme.txt)
3. python train.py

**Test**

1. Change the config 'PRETRAIN' to your pretrain model path.
2. python test.py (metrics include Accuracy, mF1, mIoU, MAE, and RMSE)

**Weights**

* Our pre-train model on Hong Kong dataset could be also downloaded from link: [download link](https://huggingface.co/datasets/littlebeen/Buildingage/tree/main)

If you have any questions, be free to contact me!
