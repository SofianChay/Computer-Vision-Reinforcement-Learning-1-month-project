from __future__ import print_function
import sys
sys.path = ['../..'] + sys.path
from DFP.multi_experiment import MultiExperiment
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import torch
import os 
from vision_utils.U_Net import UNet
import argparse
import matplotlib.pyplot as plt 

####################################################################################################################################
# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--ground_truth", default=False, help="Type --ground_truth 1 if you want to train the agent with ground truth labels", type=bool)
parser.add_argument("--train", default="train", help="Type show if you want to see the results of pretrained agent")
parser.add_argument("--vision", default=False, help="Type --vision 1 if you do not want to use vision algorithms", type=bool)
args = parser.parse_args()
#####################################################################################################################################


def main(train, ground_truth, vision):
	
	### Set all arguments
	
	## Target maker
	target_maker_args = {}
	target_maker_args['future_steps'] = [1,2,4,8,16,32]
	target_maker_args['meas_to_predict'] = [0,1,2]
	target_maker_args['min_num_targs'] = 3	
	target_maker_args['rwrd_schedule_type'] = 'exp'
	target_maker_args['gammas'] = []
	target_maker_args['invalid_targets_replacement'] = 'nan'
	
	## Simulator
	simulator_args = {}
	simulator_args['config'] = '../../maps/D3_battle.cfg'
	simulator_args['resolution'] = (160, 120)
	simulator_args['frame_skip'] = 4
	simulator_args['color_mode'] = 'GRAY'	
	simulator_args['use_shaping_reward'] = False
	simulator_args['maps'] = ['MAP01']
	simulator_args['switch_maps'] = False
	#train
	simulator_args['num_simulators'] = 8
	
	## Experience
	# Train experience
	train_experience_args = {}
	train_experience_args['memory_capacity'] = 20000
	train_experience_args['history_length'] = 1
	train_experience_args['history_step'] = 1
	train_experience_args['action_format'] = 'enumerate'
	train_experience_args['shared'] = False
	
	# Test prediction experience
	test_prediction_experience_args = train_experience_args.copy()
	test_prediction_experience_args['memory_capacity'] = 1
	
	# Test policy experience
	test_policy_experience_args = train_experience_args.copy()
	test_policy_experience_args['memory_capacity'] = 25000
		
	## Agent	
	agent_args = {}
	
	
	# preprocessing
	##########################################################################################################################################
	agent_args['preprocess_input_images'] = lambda x: x 
	##########################################################################################################################################
	agent_args['preprocess_input_measurements'] = lambda x: x / 100. - 0.5
	targ_scale_coeffs = np.expand_dims((np.expand_dims(np.array([7.5,30.,1.]),1) * np.ones((1,len(target_maker_args['future_steps'])))).flatten(),0)
	agent_args['preprocess_input_targets'] = lambda x: x / targ_scale_coeffs
	agent_args['postprocess_predictions'] = lambda x: x * targ_scale_coeffs
		
	# agent properties
	agent_args['objective_coeffs_temporal'] = [0., 0. ,0. ,0.5, 0.5, 1.]
	agent_args['objective_coeffs_meas'] = [0.5, 0.5, 1.]
	agent_args['random_exploration_schedule'] = lambda step: (0.02 + 145000. / (float(step) + 150000.))
	agent_args['new_memories_per_batch'] = 8
	
	# net parameters
	agent_args['conv_params']     = np.array([(32,8,4), (64,4,2), (64,3,1)],
									 dtype = [('out_channels',int), ('kernel',int), ('stride',int)])
	agent_args['fc_img_params']   = np.array([(512,)], dtype = [('out_dims',int)])
	agent_args['fc_meas_params']  = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)]) 
	agent_args['fc_joint_params'] = np.array([(512,), (-1,)], dtype = [('out_dims',int)]) # we put -1 here because it will be automatically replaced when creating the net
	agent_args['weight_decay'] = 0.00000
	
	# optimization parameters
	agent_args['batch_size'] = 64
	agent_args['init_learning_rate'] = 0.0001
	agent_args['lr_step_size'] = 250000
	agent_args['lr_decay_factor'] = 0.3
	agent_args['adam_beta1'] = 0.95
	agent_args['adam_epsilon'] = 1e-4		
	agent_args['optimizer'] = 'Adam'
	agent_args['reset_iter_count'] = False
	
	# directories		
	agent_args['checkpoint_dir'] = 'checkpoints'
	agent_args['log_dir'] = 'logs'
	agent_args['init_model'] = ''
	agent_args['model_name'] = "predictor.model"
	agent_args['model_dir'] = time.strftime("%Y_%m_%d_%H_%M_%S" + f"vision_{str(args.vision)}" + f"ground_truth_{str(args.ground_truth)}")			
	
	# logging and testing
	agent_args['print_err_every'] = 50
	agent_args['detailed_summary_every'] = 1000
	agent_args['test_pred_every'] = 0
	agent_args['test_policy_every'] = 100
	agent_args['num_batches_per_pred_test'] = 0
	agent_args['num_steps_per_policy_test'] = test_policy_experience_args['memory_capacity'] / simulator_args['num_simulators']
	agent_args['checkpoint_every'] = 10000
	agent_args['save_param_histograms_every'] = 5000
	agent_args['test_policy_in_the_beginning'] = True				
	
	# experiment arguments
	experiment_args = {}
	experiment_args['num_train_iterations'] = 20000
	experiment_args['test_objective_coeffs_temporal'] = np.array([0., 0., 0., 0.5, 0.5, 1.])
	experiment_args['test_objective_coeffs_meas'] = np.array([0.5,0.5,1.])
	experiment_args['test_random_prob'] = 0.
	experiment_args['test_checkpoint'] = 'checkpoints/2017_04_08_10_44_20'
	experiment_args['test_policy_num_steps'] = 3125
	experiment_args['show_predictions'] = False
	experiment_args['multiplayer'] = False 
	
	
##########################################################################################################################################################################
	agent_args['train_vision_model_every'] = experiment_args['num_train_iterations'] // 10
	agent_args['write_video_every'] =  5000
	assert(agent_args['write_video_every'] % agent_args['test_policy_every'] == 0), 'video writing must be at test time'
##########################################################################################################################################################################

##########################################################################################################################################################################
	# create the segmentation model
	model = UNet(1, 6)


	# Create and run the experiment
	
	experiment = MultiExperiment(target_maker_args=target_maker_args, 
							simulator_args=simulator_args, 
							train_experience_args=train_experience_args, 
							test_policy_experience_args=test_policy_experience_args, 
							agent_args=agent_args,
							experiment_args=experiment_args, model=model, ground_truth=ground_truth, vision=vision)
##########################################################################################################################################################################

	return experiment.run(train) 

if __name__ == '__main__':
	rwrd_dict = main(args.train, args.ground_truth, args.vision)
	plt.rc_context(rc={'axes.grid': True})
	plt.plot(np.arange(len(rwrd_dict["total_avg_rwrd"])), rwrd_dict["total_avg_rwrd"],  linestyle='solid', color='c')
	plt.title("average reward per episode")
	plt.savefig(f"vision_{args.vision}_ground_truth_{args.ground_truth}_rewards.png")
	# ammu
	plt.clf()
	plt.plot(np.arange(len(rwrd_dict["total_avg_meas"])), [meas[0] for meas in rwrd_dict["total_avg_meas"]],  linestyle='solid', color='b')
	plt.title("average munitions per episode")
	plt.savefig(f"vision_{args.vision}_ground_truth_{args.ground_truth}_ammu.png")
	# health
	plt.clf()
	plt.plot(np.arange(len(rwrd_dict["total_avg_meas"])), [meas[1] for meas in rwrd_dict["total_avg_meas"]],  linestyle='solid', color='g')
	plt.title("average health per episode")
	plt.savefig(f"vision_{args.vision}_ground_truth_{args.ground_truth}_health.png")
	# frags
	plt.clf()
	plt.plot(np.arange(len(rwrd_dict["total_avg_meas"])), [meas[2] for meas in rwrd_dict["total_avg_meas"]],  linestyle='solid', color='r')
	plt.title("average frags per episode")
	plt.savefig(f"vision_{args.vision}_ground_truth_{args.ground_truth}_frags.png")
