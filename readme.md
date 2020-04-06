1. Conv2DTranspose
    set kernel_size = 2 \* stride


2. element-wise add
    RUN 1\*1 conv on pool3 & pool4 to obtain feature maps with [num_classes] channels


3. initialization
    Final layer deconvolutional filters are fixed to bilinear interpolation, 
    while intermediate upsampling layers are initialized to bilinear upsampling, and then learned. 


4. first layer padding
    FCN的作者在第一层直接对原图加了100的padding，因为conv以后，特征图尺寸缩减为原图的1/32，
    再经过fc6的7\*7 conv，特征图尺寸变为(h/32-7)/1+1=(h-192)/32，也就是说原网络对于长宽不超过192的图片是没法处理的，
    而pad100以后，fc6的输出尺寸变为(h+8)/32
    或者减少池化层，但是这样直接就改变了原先可用的结构了，就不能用以前的结构参数进行fine-tune了


5. upSampling padding
    原文上采样采用valid padding，
    卷积过程：$W_o = (W_i + 2\*pad - kernel\_size) / stride + 1$
    逆运算：$W_o = (W_i - 1) * stride + kernel\_size - 2\*pad$
    于是32x: ((h+8)/32-1)\*32+64 = h+40，与原图尺寸不一致，需要crop
    caffe源代码中通过指定axis和offset两个参数实现：(N,C,H+40,W+40),axis=2,offset=19->(N,C,19:H+19,19:W+19)
    