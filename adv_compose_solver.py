import numpy as np
import torch
import torch.nn.functional as F

from loss import calc_segmentation_consistency



class ComposeAdversarialTransformSolver(object):
    """
    apply a chain of transformation
    """

    def __init__(self, 
                chain_of_transforms=[],divergence_types=['kl','contour'],
                divergence_weights=[1.0,0.5],use_gpu: bool = True,
                debug: bool = False,
                disable_adv_noise = False
                ):
        '''
        apply a chain of transforms

        '''
        self.chain_of_transforms=chain_of_transforms
        self.use_gpu = use_gpu
        self.debug = debug
        self.divergence_weights=divergence_weights
        self.divergence_types=divergence_types
        self.require_bi_loss = self.if_contains_geo_transform()
        self.disable_adv_noise=disable_adv_noise

    
        
    def adversarial_training(self,data,model,init_output=None,lazy_load=False,n_iter=1, optimization_mode='chain',
            optimize_flags=None,
            power_iteration=False,
            ):
        """
        given a batch of images: NCHW, and a current segmentation model
        find optimized transformations 
        return the adversarial consistency loss for network training
        Args:
            data ([torch 4d tensor]): [input images]
            model ([torch.nn.Module]): segmentation model
            init_output([torch 4d tensor],optional):network predictions on input images using the current model.Defaults to None. 
            lazy_load (bool, optional): if true, if will use previous random parameters (if have been initialized). Defaults to False.
            n_iter (int, optional): innner iterations to optimize data augmentation. Defaults to 1.
            optimization_mode (str, optional): for composed transformation, use chain or independent optimization. supports 'chain','independent',"independent_and_compose". Defaults to 'chain'.
            optimize_flags ([list of boolean], optional): [description]. Defaults to None.
            power_iteration ([list of boolean], optional): [description]. Defaults to False.
        Raises:
            NotImplementedError: [check whether the string for specifying optimization_mode is valid]

        Returns:
            dist [loss tensor]: [adv consistency loss for network regularisation]
        """
        '''
     
        '''
        ## 1. set up optimization mode for each transformation
        if optimize_flags is not None:
            assert len(self.chain_of_transforms)==len(optimize_flags), 'must specify each transform is learnable or not'
            if self.debug: print (optimize_flags)
        else:
            if n_iter==0: optimize_flags = [False] *len(self.chain_of_transforms)
            else: optimize_flags = [True] *len(self.chain_of_transforms)
        
        if self.disable_adv_noise:
            for i, (opt, tr) in enumerate(zip(optimize_flags,self.chain_of_transforms)):
                if tr.get_name()=='noise':
                    optimize_flags[i]=False

        if isinstance(power_iteration, bool):
            power_iterations=[power_iteration]*len(self.chain_of_transforms)
        elif isinstance(power_iteration,list):
            assert len(self.chain_of_transforms)==len(power_iteration), 'must specify each transform optimization mode'
            power_iterations = power_iteration

        ## 2. get reference predictions f(x)
        if init_output is None:
            init_output = self.get_init_output(data=data,model=model)

        ## 3. optimize transformation t to maxmize the difference between f(x) and f(t(x))
        self.init_random_transformation(lazy_load)
        if n_iter ==1 or n_iter>1:
            if optimization_mode == 'chain':
                optimized_transforms = self.optimizing_transform(data=data,model=model,init_output=init_output,n_iter=n_iter,power_iterations=power_iterations,optimize_flags=optimize_flags)            
            elif optimization_mode =='independent':
                optimized_transforms = self.optimizing_transform_independent(data=data,model=model,init_output=init_output,n_iter=n_iter,power_iterations=power_iterations,optimize_flags=optimize_flags)
            elif optimization_mode =='independent_and_compose':
                optimized_transforms = self.optimizing_transform_independent(data=data,model=model,init_output=init_output,n_iter=n_iter,power_iterations=power_iterations,optimize_flags=optimize_flags)
            
            else:
                raise NotImplementedError
            self.chain_of_transforms=optimized_transforms

        else:
            pass
            #print ('random')
        
        ## 4. augment data with optimized transformation t, and calc the adversarial consistency loss with the composite transformation
        if optimization_mode == 'chain':
            dist,adv_data,adv_output,warped_back_adv_output = self.calc_adv_consistency_loss(data.detach().clone(),model,init_output =init_output,chain_of_transforms=self.chain_of_transforms)
        elif optimization_mode == 'independent':
            ## augment data with each type of adv transform indepdently, the loss is defined as the average of each adv consistency loss
            dist = torch.tensor(0., device= data.device)
            for transform in self.chain_of_transforms:
                dist_i,adv_data,adv_output,warped_back_adv_output = self.calc_adv_consistency_loss(data.detach().clone(),model,init_output =init_output, 
                chain_of_transforms=[transform])
                dist+=dist_i
                if self.debug:
                    print ('{}:loss:{}'.format(transform.get_name(),dist_i))

            dist/= float(len(self.chain_of_transforms))
        elif optimization_mode == 'independent_and_compose':
            ## optimize each type of adv transform indepdently,and then compose them with a fixed order:
            dist,adv_data,adv_output,warped_back_adv_output = self.calc_adv_consistency_loss(data.detach().clone(),model,init_output =init_output,chain_of_transforms=self.chain_of_transforms)
        else:
            raise NotImplemented

 
        self.init_output = init_output
        self.warped_back_adv_output=warped_back_adv_output
        self.origin_data = data
        self.adv_data=adv_data
        self.adv_predict=adv_output
        if self.debug:
            print ('outer loop loss',dist)
            print ('init out',init_output.size())
        return dist

    def calc_adv_consistency_loss(self,data,model,init_output,chain_of_transforms=None):
        """[summary]  
        calc adversarial consistency loss with adversarial data augmentation 

        Args:
            data ([torch 4d tensor]): a batch of clean images
            model ([torch.nn.Module]):segmentation model
            init_output ([torch 4d tensor]): predictions on clean images (before softmax)
            chain_of_transforms ([list of adversarial image transformation], optional): [description].
             Defaults to None. use self.chain_of_transform

        Returns:
            loss [torch.tensor]: The consistency loss  
        """
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        adv_data = self.forward(data,chain_of_transforms)
        model.train()
        model.zero_grad()
        with torch.enable_grad():
            adv_output = model(adv_data)
       
        ## calc divergence loss in bi-directional fashion when geometric transformation is involved
        if self.if_contains_geo_transform(chain_of_transforms):
            warped_back_adv_output = self.backward(adv_output,chain_of_transforms)
            mask = torch.ones_like(adv_output,device=adv_output.device)
            with torch.no_grad():
                forward_mask=self.predict_forward(mask,chain_of_transforms)
                backward_mask =self.predict_backward(forward_mask,chain_of_transforms)
                forward_reference=self.predict_forward(init_output.detach(),chain_of_transforms)
            
            dist = 0.5*(
                     self.loss_fn(pred = warped_back_adv_output, reference = init_output.detach(), mask=backward_mask.detach())
                     +self.loss_fn(pred = adv_output, reference = forward_reference, mask=forward_mask.detach()))
          
        else:
            ## no geomtric transformation
            warped_back_adv_output= adv_output
            mask= torch.ones_like(adv_output)

            dist =  self.loss_fn(pred = adv_output, reference = init_output.detach(), mask=mask)
        ##  average the consistency loss
        dist = 1/len(chain_of_transforms)*dist
        return dist,adv_data,adv_output,warped_back_adv_output
    
    
    def forward(self, data,chain_of_transforms=None):
        '''
        forward the data to get transformed data
        :param data: input images x, NCHW
        :return:
        tensor: transformed images, NCHW
        '''
        self.diffs=[]
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for transform in chain_of_transforms:
            data = transform.forward(data)
            self.diffs.append(transform.diff)
        return data
    
    def predict_forward(self, data,chain_of_transforms=None):
        '''
        transform the prediction with the learned/random data augmentation, only applies to geomtric transformations.
        :param data: input images x, NCHW
        :return:
        tensor: transformed images, NCHW
        '''
        self.diffs=[]
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for transform in chain_of_transforms:
            data = transform.predict_forward(data)
            self.diffs.append(transform.diff)
        return data

    def loss_fn(self,pred,reference,mask=None):
        if mask is not None:
            mask [mask!=0]= 1
        scales=[0]
        loss =calc_segmentation_consistency(output=pred, reference=reference.detach(),divergence_types=self.divergence_types,divergence_weights=self.divergence_weights,scales=scales,mask=mask)
        return loss
    
    def if_contains_geo_transform(self, chain_of_transforms=None):
        """[summary]
        check if the predefined transformation contains geometric transform
        Returns:
            [boolean]: return True if geometric transformation is involved, otherwise false.
        """
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        sum_flag=0

        for transform in chain_of_transforms:
            sum_flag+= transform.is_geometric()
        return sum_flag>0



    def init_random_transformation(self,lazy_load=False):
        ## initialize transformation parameters
        '''
        randomly initialize random parameters
        return 
        list of random parameters, and list of random transform

        '''
        for transform in self.chain_of_transforms:
            if lazy_load:
                 if transform.param is None:
                     transform.init_parameters()
            else:
                transform.init_parameters()
    
    def set_transformation(self, parameter_list):
        """
        set the values of transformations accordingly

        Args:
            parameter_list ([type]): [description]
        """
        ## reset transformation parameters
        for i, param in enumerate(parameter_list):
             self.chain_of_transforms[i].set_parameters(param)

    def make_learnable_transformation(self,power_iterations,optimize_flags,chain_of_transforms=None):
        """[summary]
        make transformation parameters learnable
        Args:
            power_iterations ([boolean]): [description]
            chain_of_transforms ([list of adv transformation functions], optional): 
            [description]. Defaults to None. if not specified, use self.transformation instead
        """
        ## reset transformation parameters
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for flag, power_iteration,transform in zip(optimize_flags,power_iterations,chain_of_transforms):
             if flag: 
                if power_iteration:
                    ## for second-order based optimization, requires small transformation to approx hessian
                    transform.make_small_parameters()
                transform.train()

        
    def optimizing_transform(self, model,data,init_output,power_iterations,optimize_flags, n_iter=1):
        ## optimize each transform with one forward pass.
        model.eval()
        clean_data = data.detach().clone()
        for i in range(n_iter):
            torch.cuda.empty_cache()
            model.zero_grad()
            self.make_learnable_transformation(power_iterations=power_iterations,optimize_flags=optimize_flags,chain_of_transforms=self.chain_of_transforms)
            augmented_data = self.forward(clean_data)
            perturbed_output = model(augmented_data)
            ## calc divergence loss 
            if self.require_bi_loss:
                warped_back_prediction = self.backward(perturbed_output) 
                mask = torch.ones_like(perturbed_output,device=augmented_data.device)
                mask.requires_grad=False
                with torch.no_grad():
                    forward_mask=self.predict_forward(mask)
                    backward_mask =self.predict_backward(forward_mask)
                    forward_reference=self.predict_forward(init_output.detach())
       
                dist = 0.5*(self.loss_fn(pred = warped_back_prediction, reference =init_output.detach(),mask=backward_mask.detach())
                        +self.loss_fn(pred=perturbed_output,reference=forward_reference,mask=forward_mask.detach()))
            else:
                dist = self.loss_fn(pred = perturbed_output, reference =init_output.detach(),mask=None)

            if self.debug: print ('{} inner loop: dist {}'.format(str(i),dist.item()))
            dist.backward()

            for flag,power_iteration,transform in zip(optimize_flags,power_iterations,self.chain_of_transforms):
                if flag:
                    if self.debug: print ('update {} parameters'.format(transform.get_name()))
                  
                    transform.optimize_parameters(power_iteration=power_iteration,step_size = 1/np.sqrt((i+1)))
            model.zero_grad()
            
        transforms=[]
        for flag,power_iteration,transform in zip(optimize_flags,power_iterations,self.chain_of_transforms):
            if flag: 
                transform.rescale_parameters(power_iteration =power_iteration)
                transform.eval()
            transforms.append(transform)
        model.train()
        return transforms
    
    
    def optimizing_transform_independent(self,data,model,init_output,power_iterations,optimize_flags,lazy_load=False,n_iter=1):
        ## optimize each transform individually.
        model.eval()
        new_transforms = []
        for opti_flag, power_iteration,transform in zip(optimize_flags,power_iterations,self.chain_of_transforms):
            torch.cuda.empty_cache()

            if opti_flag:                
                for i in range(n_iter):
                    torch.cuda.empty_cache()
                    self.make_learnable_transformation([power_iteration],chain_of_transforms=[transform])
                    augmented_data = transform.forward(data) 
                    # with _disable_tracking_bn_stats(model):
                    perturbed_output = model(augmented_data)
                    if transform.is_geometric()>0:
                        warped_back_prediction = transform.backward(perturbed_output) 
                        mask = torch.ones_like(perturbed_output,device=augmented_data.device)
                        mask.requires_grad=False
                        with torch.no_grad():
                            forward_mask=transform.predict_forward(mask)
                            backward_mask =transform.predict_backward(forward_mask)
                            forward_reference=transform.predict_forward(init_output.detach())
       
                        dist = 0.5*(self.loss_fn(pred = warped_back_prediction, reference =init_output.detach(),mask=backward_mask.detach())
                               +self.loss_fn(pred=perturbed_output,reference=forward_reference,mask=forward_mask.detach()))
                    else:
                        dist = self.loss_fn(pred = perturbed_output, reference =init_output.detach(),mask=None)
                    if self.debug: print ('{} dist {} '.format(str(i),dist.item()))
                    dist.backward()
                    transform.optimize_parameters(power_iteration=power_iteration,step_size = 1/np.sqrt(i+1))
                    model.zero_grad()
                transform.rescale_parameters(power_iteration=power_iteration)
                transform.eval()
            new_transforms.append(transform)
        model.train()
        return new_transforms
    
    
    def get_init_output(self,model,data):
        model.eval()
        with torch.no_grad():
            reference_output = model(data)
        model.train()
        return reference_output
    
    def backward(self,data,chain_of_transforms=None):
        '''
        warp it back to image space
        only activate when the augmentation is a geometric transformation
        '''
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for transform in reversed(chain_of_transforms):
            data = transform.backward(data)
        return data

    def predict_backward(self,data,chain_of_transforms=None):
        '''
        warp it back to image space
        only activate when the augmentation is a geometric transformation
        '''
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for transform in reversed(chain_of_transforms):
            data = transform.predict_backward(data)
        return data




if __name__ == "__main__":
    import os
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import seaborn as sns
    from skimage import data
    import SimpleITK as sitk
    import numpy as np
    from utils import check_dir

    log_dir = "./result/log/debug/"
    check_dir(log_dir, create=True)
    # from models.image_transformer.adv_affine import AdvAffine 
    # from models.image_transformer.adv_noise import AdvNoise
    # from models.image_transformer.adv_morph import AdvMorph
    from adv_bias import AdvBias
    # from models.model_util import get_unet_model

    sns.set(font_scale=1)

    ## 1. set up image tensor
    image_size= (1,1,128,128)
    ### test random aug
    images = torch.randn(image_size).cuda()
    print ('input:',images)

    sample_image_path='/vol/biomedic3/cc215/Project/DeformADA/Data/ACDC_small/ES/003_img.nrrd'
    sample_image = sitk.GetArrayFromImage(sitk.ReadImage(sample_image_path))[np.newaxis,:,:,:][:,[6],:,:]
    sample_image = (sample_image-sample_image.min())/(sample_image.max()-sample_image.min())
    sample_image_tensor= torch.from_numpy(sample_image).float()
    images[:,[0]]=sample_image_tensor.cuda()
    images = images.cuda()
    images.requires_grad=False
    print ('input',images)
    ## 2. set up data augmentation and its optimizer
    augmentor_bias= AdvBias(
                 config_dict={'epsilon':0.3,
                 'xi':0.1,
                 'control_point_spacing':[28,28],
                 'downscale':2,
                 'data_size':image_size,
                 'interpolation_order':3,
                 'init_mode':'gaussian',
                 'space':'log'},debug=True)

    chain_of_transforms=[augmentor_bias]

    ## optimizer
    power_iteration=False
    n_iter= 1
    composed_augmentor = ComposeAdversarialTransformSolver(
        chain_of_transforms=chain_of_transforms,
        divergence_types = ['kl','contour'],
        divergence_weights=[1.0,0.5],
        use_gpu= True,
        debug=True,
        disable_adv_noise=True)

    ## 3. set up  the segmentor
    model = torch.nn.Conv2d(1,4,3,1,1)
    # model = get_unet_model(num_classes=4,model_path='./result/UNet_16$SAX$_Segmentation.pth',model_arch='UNet_16')
    model.cuda()

    ## 4. start learning
    composed_augmentor.init_random_transformation()

    ## 4.1 get randomly augmented results for reference
    rand_transformed_image = composed_augmentor.forward(images)
    rand_predict = model.forward(rand_transformed_image)
    # rand_predict = F.softmax(rand_predict,dim=1)
    model.zero_grad()
    rand_recovered_predict = composed_augmentor.predict_backward(rand_predict)
    rand_recovered_image = composed_augmentor.backward(rand_transformed_image)
    diff = rand_recovered_image-images

    print ('sum image diff', torch.sum(diff))

    ## 4.2 adv data augmentation 
    loss = composed_augmentor.adversarial_training(
        data=images,model=model,
        n_iter=n_iter,lazy_load=True,optimization_mode='chain',
        optimize_flags=[True]*len(chain_of_transforms),
        power_iteration=[power_iteration]*len(chain_of_transforms))
    print ('consistency loss',loss.item())

    warped_back_adv_image = composed_augmentor.backward(composed_augmentor.adv_data)
    adv_predict = composed_augmentor.adv_predict
    adv_recovered_predict = composed_augmentor.warped_back_adv_output
    init_output = composed_augmentor.init_output


    fig, axes = plt.subplots(2,8)
    
    axes[0,0].imshow(images.detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[0,0].set_title('Input')

    axes[0,1].imshow(composed_augmentor.adv_data.detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[0,1].set_title('Transformed')

    axes[0,2].imshow(warped_back_adv_image.detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[0,2].set_title('Recovered')

  
    axes[0,3].imshow(torch.argmax(adv_predict, dim=1).unsqueeze(1).detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[0,3].set_title('Adv Predict')
    
    axes[0,4].imshow(torch.argmax(adv_recovered_predict, dim=1).unsqueeze(1).detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[0,4].set_title('Recovered')

    axes[0,5].imshow(torch.argmax(init_output, dim=1).unsqueeze(1).detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[0,5].set_title('Original')

    axes[0,6].imshow((composed_augmentor.adv_data-images).detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[0,6].set_title('diff')



    axes[1,0].imshow(images.detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[1,0].set_title('Input')

    axes[1,1].imshow(rand_transformed_image.detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[1,1].set_title('Rand ')

    axes[1,2].imshow(rand_recovered_image.detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    axes[1,2].set_title('Rand')
     
    axes[1,3].imshow(torch.argmax(rand_predict, dim=1).detach().cpu().numpy()[0],cmap='gray',interpolation=None)
    axes[1,3].set_title('Predict')
    axes[1,4].imshow(torch.argmax(rand_recovered_predict, dim=1).detach().cpu().numpy()[0],cmap='gray',interpolation=None)
    axes[1,4].set_title('Recovered')
    axes[1,5].imshow(torch.argmax(init_output, dim=1).detach().cpu().numpy()[0],cmap='gray',interpolation=None)
    axes[1,5].set_title('Original')

    axes[1,6].imshow((rand_transformed_image-images).detach().cpu().numpy()[0,0],cmap='gray',interpolation=None)
    

    for ax in axes.ravel():
        ax.set_axis_off()
        ax.grid(False)
    plt.tight_layout(w_pad=0,h_pad=0)
    save_dir = './result/log/debug/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,'test_noise_affine_morph_adv.png'))
    plt.clf()



   

    