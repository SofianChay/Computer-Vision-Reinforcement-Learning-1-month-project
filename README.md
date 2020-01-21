
# Computer Vision / Reinforcement Learning (1-month) project 


The aim of my work is to improve the reinforcement learning agent from [[1]](https://arxiv.org/pdf/1611.01779.pdf) which achieves state-of-the-art results in learning **sensorimotor control** from raw sensory input in complex and dynamic three-dimensional environments, learned directly from experience. The improvement is done by incorporating additional information to the agent such as segmentation of objects or depth detection of the raw image.  
The experiments are conducted on the FPS game **Doom** through the AI Researh Platform for RL [**ViZDoom**](http://vizdoom.cs.put.edu.pl/). 

>Agent trained to play ViZDoom on a particular environment helped by segmentation and depth detection.
![outputvideo_30000_vision_True_groundtruth_True](https://user-images.githubusercontent.com/58939729/72359518-b54c1500-36ee-11ea-8a0c-d346ce4f5947.gif)
![outputvideo_depth_30000_vision_True_groundtruth_True](https://user-images.githubusercontent.com/58939729/72359519-b54c1500-36ee-11ea-81ff-6c2528edf551.gif)
![outputvideo_segmentation_30000_vision_True_groundtruth_True](https://user-images.githubusercontent.com/58939729/72359520-b5e4ab80-36ee-11ea-9e3c-0e5cde6b6999.gif)

## Aknowledgements 

Before  beginning  the  description  of  the  core  issues,  I have to acknowledge the work of [3].  In this paper, similar experiments are described and the main ideas inspired me and helped me to understand the problematic and what I had to do to solve it.


# Standard Reinforcement Learning Approach vs Direct Future Prediction Approach

In this section, we briefly develop the approach taken by [1] to address learning sensorimotor control problem.

## Standard Reinforcement Learning 
![ Standard RL](https://user-images.githubusercontent.com/58939729/72206076-fee7f600-3489-11ea-909b-1cb3c5b10095.png)  

The usual paradigm in RL is composed of an **agent** (e.g : player) interacting with an **environment** (e.g : Doom map). The agent is able to perform a known set of **actions** (e.g : shoot, go left, go right, ..). The interaction between the agent and the environment is materialized by a (stochastic) **reward** (e.g : +1 for each new time step the agent lives) and an **observation** (e.g : the raw image of the game) which will be useful to perform the next action.  

> The goal of this approach is to **maximize the (expected) sum of future rewards.**

This approach might not be adapted to learning sensorimotor control from raw sensory input in three-dimensional environments. Indeed scalar rewards are unable to describe completely multi-dimensional environments. Instead of this, we should rather focus on the **measurements** provided to the agents (e.g : the number of frags and the health).

## Direct Future Prediction
> Figure taken from the presentation of [1] at ICLR 2017
![Direct Future Prediction](https://user-images.githubusercontent.com/58939729/72206075-fee7f600-3489-11ea-86a3-0a21df49528a.png) 

The approach developed by [1] is **Future supervised learning**. The goal is to predict measurements available to the agent. For this purpose, it is assumed that the rewards can be expressed as (linear) function of measurements. In this configuration, the role of the agent is to predict the future measurements (at multiple future time steps) implied by each possible action based on raw input, and present measurements. From each future measurements, the expected reward is deduced.   

![Measurements available to the agent (Ammo, Health, ...)](https://user-images.githubusercontent.com/58939729/72206120-6605aa80-348a-11ea-91e2-994ec9b3a91f.png)

Eventually, the main goal of this approach is to learn a predictor function, i.e : a function able to predict future measurements from raw input and present measurements for each possible actions. This function is approximated by a neural network. 
>  [[1]](https://arxiv.org/pdf/1611.01779.pdf)  to see the details of the neural network, the results and comparisons to other methods (e.g : DQN and A3C)

# What if explicit intermediate representations are added ?

In this section, we introduce visual tools which will provide intermediate representations to the DFP agent in order to help it choose the best actions.  

The **main focus** of my project is to find out if providing intermediate representations to the DFP agent will help it learning better policies. For this purpose we will focus on semantic segmentation and depth detection. 

## Segmentation

Semantic segmentation is the partition of an image into coherent parts. For example classifying each pixel that belongs to a person, a car, a tree or any other entity in a dataset.  

> Here is an example of semantic segmentation provided by the ViZDoom environment :  
![segmentation exemple](https://user-images.githubusercontent.com/58939729/72206455-6dc74e00-348e-11ea-9201-87b62b0fbbd4.png)

## Depth Detection

Depth can be defined as the distance from the camera for each pixel in the image frame.  

>Here is a color map of depth provided by the ViZDoom environment.  
![depth example](https://user-images.githubusercontent.com/58939729/72206454-6dc74e00-348e-11ea-8c1e-6dafd1772289.png)

## U-Net [[2]](https://arxiv.org/pdf/1505.04597.pdf) : An ecoder-decoder architecture to predict intermediate representations 

To automate segmentation and depth detection of frames, we aim to train a neural network. As it is done in [3] The architecture I use is U-Net [2]. The architecture contains two paths. First path is the encoder which is used to capture the context in the image. The second path is the decoder which is used to enable precise localization. Originally, the output tensor of U-Net has lower height and width than the input image yet we need the output to have the same dimensions in order to be concatenated to the raw image and provided to the DFP agent. Thus, I modified the original architecture for this purpose. 

> U-Net architecture  
![Unet](https://user-images.githubusercontent.com/58939729/72361960-9cddf980-36f2-11ea-8361-9bf9b79963d1.png)

### Qualitative results of U-Net 
In this subsection, I present qualitative results of U-Net on segmentation task and depth detection. For these tests, very few train images (2000 examples) were used and the training was done on very few epochs (4 epochs). It shows the efficiency of this model since excellent results can be observed. The optimizer used is stochastic gradient descent. The implementation is done in PyTorch.

#### Segmentation

For semantic segmentation. The goal of the model is to predict the category of each pixel. The categories considered in this project are : **WALL**, **CEILING**, **FLOOR**, **ENEMY**, **ITEM**, **OTHER**. For each category and for each pixel a probability that the pixel belongs to the category is predicted. Thus, the cross-entropy loss is minimized.  

```
Input 					Prediction 				Ground Truth
``` 
![prediction_segmentation](https://user-images.githubusercontent.com/58939729/72361959-9c456300-36f2-11ea-8e5d-ae3c5ce14ae5.png)

#### Depth Detection 

For depth detection. The goal is to predict the distance of the object represented in a pixel to the doom player. Thus, the mean square error was minimized.  

```
Input 					Prediction 				Ground Truth
``` 

![depth_detection_prediction](https://user-images.githubusercontent.com/58939729/72361958-9c456300-36f2-11ea-8b89-b6174624dea9.png)

# Structure of the model 

Eventually, based on U-Net architecture and DFP agent, we build a model that takes as input the raw image, the present measurements and the goal, to predict the future measurements and to take action, exactly as the DFP agent. The main difference is the intermediate step, composed of a segmentation predictor and depth predictor which takes the raw image as input and outputs its segmentation and depth representations. All these three images are concatenated and provided to the DFP agent. 

![architecture](https://user-images.githubusercontent.com/58939729/72374183-0ddcdb80-370a-11ea-9f72-bfcbfc8df6c1.png)


# Structure of the code 



## Requirements 

* python3.6
* tensorflow==2.0.0  
> In fact the authors of [1] used tensorflow1 but for Cuda compatibility issues I use tensorflow2 and in the code I replace the following line :  

```
import tensorflow as tf 
```

>by  

```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
``` 

* pytorch==1.3.1
* scikit-video==1.1.11
* numpy==1.17.4


## Run Experiments 

> To run the code :
```
$ cd examples/D3_battle 
```  
> You can change ```D3_battle``` to any other folder in examples (other configurations, see [1] for more details.)
> To run without any intermediate representations 
```
$ python run_exp.py
```  
> To run with predicted intermediate representations 
```
$ python run_exp.py --vision 1 
```

> To run with groundtruth intermediate representations 
```
$ python run_exp.py --vision 1 --groundtruth 1
```
## Architecture 

I reuse the code provided with [1] I retrieve the DFP agent provided and I modify it in order to integrate the computer vision models and the intermediate representations. The architecture of the code is mainly composed of the DFP agent interacted with a memory. The memory is composed of a certain amount of states and the groundtruth predictions. It feeds the agent which learns the policy. Conversely, the agent takes actions according an epsilon-greedy policy (epsilon decreasing with time) in order to provide new states to the memory.

### Handling the "covariate shift in behavioral cloning"

As the agent is learning and its policy is improving, the agent is likely to discover new states and the distribution of the states might be very different over time. It is a problem for the computer vision trained model. Indeed, if they are trained only at the begining, they will very likely be wrong on never seen states. It is known as the "covariate shift in behavioral cloning". To handle this problem, I have decided to interupt the learning of the agent at regular times for the depth model and the segmentation model to be trained on the new states.

# Conditions of Experiments 

In this section, conditions and parameters of the experiments are described. The experiments were conducted on the Doom environment D3 from [1].  

## Doom environment 
The agent has a first person view of the environment and act on the raw image that is shown to human players (with potentially visual tools). In this scenario the goal of the player is to defend against attacking ennemies. Health and ammunition can be gathered. The measurements available to the agent are : health, frags and ammunition. The possible actions are :  move forward, move backward, turn left, turn right, strafe left, strafe right, run, and shoot. Any combination of actions is also an action.

> Doom environment :  
![doom_env](https://user-images.githubusercontent.com/58939729/72364641-20014e80-36f7-11ea-94fd-06751eae86e1.png)


## Experiments 

In the orginal paper, the agent is trained over 800,000 mini-batch iterations. For a matter of ressources I reduce the training to 50,000 mini-batch iterations. To retrieve examples, the agent runs experiment and chooses action following an epsilon-greedy policy with the parameter epsilon 
decreasing with the training step. To speed up convergence, I make the parameter epsilon decrease faster so that exploration is reduced. Eventually, the policy obtained is certainly sub-optimal. Nevertheless, all experiments are conducted in the exact same conditions so it is a good basis to evaluate the improvements.  

Three different agents are trained in the conditions described above. The first one is trained without any intermediate representations. The second one is trained with groundtruth segmentation and depth. The last one is trained with Unet predictions of segmentation and depth.


## Evaluation

The evaluation of the agent is done multiple time during training : each 1000 mini-batch iterations, the policy learned (i.e the predictor function associated to the greedy policy) is tested over 20000 steps (A step corresponds to one action). After that, the mean measurements and rewards over these episodes are provided.

## Parameters 

The following table summurizes the parameters of the experiments :  

| Parameter            | Value                 |
|:--------------------:|:---------------------:|
|future steps predicted|[1,2,4,8,16,32]        |
|measure to predict    |Frags, health, ammo    |
|resolution of the raw input|(160px, 120px)      |
|training memory capacity | 20000      |
|New steps per iteration| 64|
|batch size |64      |
|policy tested every| 1000 mini-batch iterations      |
|mini-batch iterations| 50000|
|test number of steps| 20000|
|vision models updating (if used) | 10 times|



# Results

## Results for the agent trained without intermediate representations 
![vision_False_ground_truth_False_frags](https://user-images.githubusercontent.com/58939729/72359729-17a51580-36ef-11ea-9f98-519c1b689e2f.png)


## Results for the agent trained with groundtruth depth detection and segmentation
![vision_True_ground_truth_True_frags](https://user-images.githubusercontent.com/58939729/72359749-2095e700-36ef-11ea-8f68-59066e5e3038.png)


## Results for the agent trained with predicted depth detection and segmentation
![vision_True_ground_truth_False_frags](https://user-images.githubusercontent.com/58939729/72765877-e3e65600-3bee-11ea-9892-32bf87a0d784.png)

## Result combined 
![final_results_without_std](https://user-images.githubusercontent.com/58939729/72814170-e2f00b80-3c64-11ea-9851-f3f80ed9b946.png)
![final_results_with_std](https://user-images.githubusercontent.com/58939729/72814171-e2f00b80-3c64-11ea-942d-65bb9e6b7213.png)


## Observations 

We can make several comments from the observation of the figures above. Comparing the curves of reward and frags with and without intermediate representations, we notice several characteristics that allow to think the initial model is improved thanks to depth detection and segmentation :   
* A better optimum is reached in both reward and frags when using vision.
* Learning is faster : the optimum is reached sooner when using vision.
* Learning is more stable : we observe less oscillation when using vision. Learning is smoother.

## Summary :
The following table summurizes the quantitative results of the experiments :  

|                      | Frags                 | Rewards                 |
|:--------------------:|:---------------------:|:-----------------------:|
|No vision             |Max : 10.5; Mean : 6.75|Max : 18; Mean : 11.25   |
|Vision and Groundtruth|Max : 12; Mean : 8.20  |Max : 20.25; Mean : 14.05|
|Vision and predictions|Max : 13.20; Mean : ?      |Max : 21.10; Mean : ?        |



# Conclusion, critics and future work
* Computer vision does improve learning **sensorimotor control** from raw sensory input in complex and dynamic three-dimensional environments. Indeed, we have observed that these intermediate representations have multiple benefits on the agent : particularly, a better and more stable learning. 

## Critics
* The agent is significantly slower (one iteration takes more time) to learn when using vision.
* It is also slower to make decision : it might be a problem when it is used in real life speed.

## future works
* Improve the vision models : for instance, try to fine-tune pre-trained models.
* **Test the power of generalisation of this model : try other maps (change textures for instance)** 
* Apply to other tasks : end-to-end learning for autonomous driving or navigation in indoor environments.

# References 
[1] [Learning to Act by Predicting the Future, Alexey Dosovitskiy and Vladlen Koltun, International Conference on Learning Representations (ICLR), 2017](https://arxiv.org/pdf/1611.01779.pdf)  
[2] [O. Ronneberger, P. Fischer, T. Brox, U-net : Convolutional networks forbiomedical image segmentation,MICCAI(2015).](https://arxiv.org/pdf/1505.04597.pdf)  
[3] [Does Computer Vision Matter for Action ? , Brady Zhou, Philipp Krahenbühl, AND Vladlen Koltun](http://vladlen.info/papers/does-vision-matter.pdf)   
