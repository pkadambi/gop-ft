import sys
import yaml
import argparse
import os
from IPython import embed

from src.ExperimentStages import *

def generate_arguments(args_dict):
	res = ""
	for arg_name, value in args_dict.items():
		res = res + "--" + arg_name + " " + str(value) + " "
	return res

def run_script(script, args_dict):
	arguments = generate_arguments(args_dict)
	return os.system("python " + script + " " + arguments)

def get_phone_count(phones_list_path):
	phones_list_fh = open(phones_list_path)
	phone_count = 0
	for line in phones_list_fh.readlines():
		line = line.split()
		use_current_phone = int(line[2])
		phone_count += use_current_phone
	return phone_count

def get_run_name(config_yaml, speakerid, use_heldout=False):
	heldout_suffix = ''
	if use_heldout:
		heldout_suffix = '_heldout'
	pathname = os.path.basename(config_yaml).split('.')[0] + heldout_suffix
	# return os.path.join(pathname)
	return pathname

def get_experiment_directory(config_yaml, speakerid, use_heldout=False):
	speaker_exp_dir = os.path.join('./experiments', speakerid)
	if not os.path.exists(speaker_exp_dir):
		os.makedirs(speaker_exp_dir, exist_ok=True)
	return os.path.join(speaker_exp_dir, get_run_name(config_yaml, speakerid, use_heldout))

def swa_identifier(is_swa):
	if is_swa:
		swa_identifier = '_swa'
	else:
		swa_identifier = ''

	return swa_identifier

def get_eval_filename(epoch, is_swa):
	swa_id = swa_identifier(is_swa)
	return "data_for_eval_epoch" + str(epoch) + swa_id + ".pickle"

def fold_identifier(use_heldout, fold_number):
	if use_heldout:
		fold_identifier = ''
	else:
		fold_identifier = '-fold-' + str(fold_number)
		
	return fold_identifier


def get_model_name(config_dict, fold, epoch=None, use_heldout=False, swa=False):
	if epoch == None:
		epoch  = config_dict["epochs"]
	run_name   = config_dict["run-name"]
	
	swa_id  = swa_identifier(swa)
	fold_id = fold_identifier(use_heldout, fold)

	return run_name +  fold_id + '-epoch-' + str(epoch) + swa_id #Aca hay codigo repetido entre el PATH de train y esto

def get_test_sample_list_path_for_fold(test_sample_list_dir, fold):
	return os.path.join(test_sample_list_dir, "test_sample_list_fold_" + str(fold)) #Aca tmb codigo repetido

def get_eval_stage(config_dict, epoch, is_swa=False):
    if config_dict.get("held-out"):
        return EvaluateScoresHeldoutStage(config_dict, epoch=epoch, is_swa=is_swa)
    else:
        return EvaluateScoresCrossValStage(config_dict, epoch=epoch, is_swa=is_swa)

def add_data_keys_to_config_dict(config_dict, setup, speakerid):

	if setup == "dataprep":
		data_dir_key = "output-dir"
		config_dict["ref-labels-dir-path"]   = config_dict["data-root-path"]
	else:
		data_dir_key = "data-dir"
	config_dict[data_dir_key] = os.path.join(config_dict[data_dir_key], speakerid)
	data_path = config_dict[data_dir_key]

	config_dict["alignments-dir-path"]   = os.path.join(data_path, "alignments")
	config_dict["alignments-path"]       = os.path.join(config_dict["alignments-dir-path"], "align_output")
	config_dict["heldout-align-path"]    = os.path.join(config_dict["alignments-dir-path"], "align_output_heldout")
	config_dict["loglikes-path"]         = os.path.join(config_dict["alignments-dir-path"], "loglikes.ark")
	config_dict["loglikes-heldout-path"] = os.path.join(config_dict["alignments-dir-path"], "loglikes_heldout.ark")
	config_dict["acoustic-model-path"]   = os.path.join(config_dict['pytorch-models-path'], 'acoustic_model.pt')
	config_dict["acoustic-model-path"]   = os.path.join(data_path, 'pytorch_models/acoustic_model.pt')
	config_dict["features-path"]         = os.path.join(data_path , "features/data")
	config_dict["features-conf-path"]    = "./data/features/conf"
	config_dict["auto-labels-dir-path"]  = os.path.join(data_path , "kaldi_labels/")
	config_dict["utterance-list-path"]   = os.path.join('./configs/speakers', speakerid, "train_full_path_list.txt")
	config_dict["train-list-path"]       = os.path.join('./configs/speakers', speakerid, "train_full_path_list.txt")
	config_dict["test-list-path"]        = os.path.join('./configs/speakers', speakerid, "heldout_full_path_list.txt")
	config_dict["reference-trans-path"]  = "./child_speech_16_khz_test/reference_transcriptions.txt"
	config_dict['speaker-id'] = speakerid

	return config_dict

def get_eval_stage(config_dict, epoch="", is_swa=False):
    if config_dict.get("held-out"):
        return EvaluateScoresHeldoutStage(config_dict, epoch=epoch, is_swa=is_swa)
    else:
        return EvaluateScoresCrossValStage(config_dict, epoch=epoch, is_swa=is_swa)