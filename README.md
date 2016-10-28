# Dynamic-Network-Surgery
Dynamic network surgery is a very effective method for DNN compression. To use it, you will need a [classic version](https://github.com/BVLC/caffe/tree/0dfc5dac3d8bf17f833e21ae6ce7bc3ea19a03fa) of [Caffe](http://caffe.berkeleyvision.org) framework.
For the convolutional and fully-connected layers to be pruned, change their layer types to "CConvolution" and "CInnerProduct" respectively. Then, pass the "cconvlution_param" and "cinner_product_param" messages to these modified layers for better pruning performance.

Please cite our work in your publications if it helps your research:

    @article{guo2016dynamic,
      title={Dynamic Network Surgery for Efficient DNNs},
      author={Guo, Yiwen and Yao, Anbang and Chen, Yurong},
      journal={arXiv preprint arXiv:1608.04493},
      year={2016}
    }

Enjoy your own surgeries!
