1. Conv2DTranspose
    set kernel_size = 2 \* stride


2. element-wise add
    2.1 1\*1 conv on pool3 & pool4
    2.2 padding on pool3 & pool4
    2.3 QUESTION: what's the difference