1. Conv2DTranspose
    set kernel_size = 2 \* stride


2. element-wise add
    RUN 1\*1 conv on pool3 & pool4 to obtain feature maps with [num_classes] channels


3. initialization
    Final layer deconvolutional filters are fixed to bilinear interpolation, 
    while intermediate upsampling layers are initialized to bilinear upsampling, and then learned. 