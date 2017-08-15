# Dynamic network surgery

Dynamic network surgery is a very effective method for DNN pruning. To better use it with python and matlab, you may also need a [classic version](https://github.com/BVLC/caffe/tree/aa2a6f55b9e50b29d607aaee0fae19bd085d6565) of the [Caffe framework](http://caffe.berkeleyvision.org).
For the convolutional and fully-connected layers to be pruned, change their layer types to "CConvolution" and "CInnerProduct" respectively. Then, pass "cconvolution_param" and "cinner_product_param" messages to these modified layers for better pruning performance. 

# Example for usage

Below is an example for pruning the "ip1" layer in LeNet5:

    layer {
      name: "ip1"
      type: "CInnerProduct"
      bottom: "pool2"
      top: "ip1"
      param {
        lr_mult: 1
      }
      param {
        lr_mult: 2
      }
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
        bias_filler {
          type: "constant"
        }
      }
      cinner_product_param {
        gamma: 0.0001
        power: 1
        c_rate: 4
        iter_stop: 14000  
        weight_mask_filler {
          type: "constant"
          value: 1
        }
        bias_mask_filler {
          type: "constant"
          value: 1
        }        
      }   
    }

# Citation

Please cite our work in your publications if it helps your research:

    @inproceedings{guo2016dynamic,		
      title = {Dynamic Network Surgery for Efficient DNNs},
      author = {Guo, Yiwen and Yao, Anbang and Chen, Yurong},
      booktitle = {Advances in neural information processing systems (NIPS)},
      year = {2016}
    } 
		
and	do not forget about Caffe:	

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

Enjoy your own surgeries!
