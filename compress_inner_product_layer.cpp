#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <cmath>

namespace caffe {

template <typename Dtype>
void CInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (this->bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
  /************ For dynamic network surgery ***************/
	CInnerProductParameter cinner_param = this->layer_param_.cinner_product_param();
	
  if(this->blobs_.size()==2 && (this->bias_term_)){
    this->blobs_.resize(4);
    // Intialize and fill the weightmask & biasmask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        cinner_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[2].get());
    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        cinner_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[3].get());    
  }  
  else if(this->blobs_.size()==1 && (!this->bias_term_)){
    this->blobs_.resize(2);	  
    // Intialize and fill the weightmask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        cinner_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[1].get());      
  }   
   
  // Intialize the tmp tensor 
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
  this->bias_tmp_.Reshape(this->blobs_[1]->shape());  

  // Intialize the hyper-parameters
  this->std = 0;this->mu = 0;  
  this->gamma = cinner_param.gamma(); 
  this->power = cinner_param.power();
  this->crate = cinner_param.c_rate();  
  this->iter_stop_ = cinner_param.iter_stop();    
  /********************************************************/
}

template <typename Dtype>
void CInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void CInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* weight = this->blobs_[0]->mutable_cpu_data();    
  Dtype* weightMask = this->blobs_[2]->mutable_cpu_data(); 
  Dtype* weightTmp = this->weight_tmp_.mutable_cpu_data(); 
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;  
  Dtype* biasTmp = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->mutable_cpu_data(); 
    biasMask = this->blobs_[3]->mutable_cpu_data();
    biasTmp = this->bias_tmp_.mutable_cpu_data();
  }
   
  if (this->phase_ == TRAIN){
    // Calculate the mean and standard deviation of learnable parameters 
		if (this->std==0 && this->iter_==0){      
			unsigned int ncount = 0;
			for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
				this->mu  += fabs(weight[k]);       
				this->std += weight[k]*weight[k];
				if (weight[k]!=0) ncount++;
			}
			if (this->bias_term_) {
				for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
					this->mu  += fabs(bias[k]);
					this->std += bias[k]*bias[k];
					if (bias[k]!=0) ncount++;
				}       
			}
			this->mu /= ncount; this->std -= ncount*mu*mu; 
			this->std /= ncount; this->std = sqrt(std);
			LOG(INFO)<<mu<<"  "<<std<<"  "<<ncount<<"\n";        
		}  		
		
		// Demonstrate the sparsity of compressed fully-connected layer
		/********************************************************/
		/*if(this->iter_%100==0){
			unsigned int ncount = 0;
			for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
				if (weightMask[k]*weight[k]!=0) ncount++;
			}
			if (this->bias_term_) {
				for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
					if (biasMask[k]*bias[k]!=0) ncount++;
				}       
			}
			LOG(INFO)<<ncount<<"\n";  			
		}*/	
		/********************************************************/	
		
		// Calculate the weight mask and bias mask with probability
		Dtype r = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
		if (pow(1+(this->gamma)*(this->iter_),-(this->power))>r && (this->iter_)<(this->iter_stop_)) { 	
			for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
				if (weightMask[k]==1 && fabs(weight[k])<=0.9*std::max(mu+crate*std,Dtype(0)))
					weightMask[k] = 0;
				else if (weightMask[k]==0 && fabs(weight[k])>1.1*std::max(mu+crate*std,Dtype(0)))
					weightMask[k] = 1;
			}	
			if (this->bias_term_) {       
				for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
					if (biasMask[k]==1 && fabs(bias[k])<=0.9*std::max(mu+crate*std,Dtype(0)))
						biasMask[k] = 0;
					else if (biasMask[k]==0 && fabs(bias[k])>1.1*std::max(mu+crate*std,Dtype(0)))
						biasMask[k] = 1;
				}    
			} 
		}
	}    
	
  // Calculate the current (masked) weight and bias
	for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
		weightTmp[k] = weight[k]*weightMask[k];
	}
	if (this->bias_term_){
		for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
			biasTmp[k] = bias[k]*biasMask[k];
		}
	} 
	
	// Forward calculation with (masked) weight and bias 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weightTmp, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(), biasTmp, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void CInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {  
	// Use the masked weight to propagate back
  const Dtype* top_diff = top[0]->cpu_diff();
  if (this->param_propagate_down_[0]) {
		const Dtype* weightMask = this->blobs_[2]->cpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();		      
    const Dtype* bottom_data = bottom[0]->cpu_data();    
    // Gradient with respect to weight
		for (unsigned int k = 0;k < this->blobs_[0]->count(); ++k) {
			weight_diff[k] = weight_diff[k]*weightMask[k];
		}		
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., weight_diff);    
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
		const Dtype* biasMask = this->blobs_[3]->cpu_data();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();		
    // Gradient with respect to bias
		for (unsigned int k = 0;k < this->blobs_[1]->count(); ++k) {
			bias_diff[k] = bias_diff[k]*biasMask[k];
		}		
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1., bias_diff);
  }
  if (propagate_down[0]) {
    const	Dtype* weightTmp = this->weight_tmp_.cpu_data();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weightTmp, (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(CInnerProductLayer);
#endif

INSTANTIATE_CLASS(CInnerProductLayer);
REGISTER_LAYER_CLASS(CInnerProduct);

}  // namespace caffe
