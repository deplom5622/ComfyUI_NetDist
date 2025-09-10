import os
import time
import json
import torch
import random
import requests
import numpy as np
from PIL import Image
from copy import deepcopy

from .utils import clean_url, get_client_id

def clear_remote_queue(remote_url):
	r = requests.get(f"{remote_url}/queue", timeout=4)
	r.raise_for_status()
	queue = r.json()

	to_cancel = []
	client_id = get_client_id()
	for k in queue.get("queue_pending", []):
		if k[3].get("client_id") == client_id:
			to_cancel.append(k[1]) # job UUID
	r = requests.post(
		f"{remote_url}/queue",
		json    = {"delete" : to_cancel},
		timeout = 4,
	)
	r.raise_for_status()

	for k in queue.get("queue_running", []):
		if k[3].get("client_id") == client_id:
			r = requests.post(
				f"{remote_url}/interrupt",
				json    = {},
				timeout = 4,
			)
			r.raise_for_status()
			break

def get_remote_os(remote_url):
	url = f"{remote_url}/system_stats"
	r = requests.get(url, timeout=4)
	r.raise_for_status()
	data = r.json()
	return data["system"]["os"]

def get_output_nodes(remote_url):
	# I'm 90% sure this could just use the
	# list from the host but better safe than sorry
	url = f"{remote_url}/object_info"
	r = requests.get(url, timeout=4)
	r.raise_for_status()
	data = r.json()
	out = [k for k, v in data.items() if v.get("output_node")]
	return out

def dispatch_to_remote(remote_url, prompt_origin, job_id=f"{get_client_id()}-unknown", outputs="final_image"):
	prompt = deepcopy(prompt_origin)
	to_del = []
	def recursive_node_deletion(start_node):
		if start_node in to_del:
			return
		
		to_del.append(start_node)
		
		inputs = prompt[start_node].get("inputs")
		if not inputs:
			return
		for iv in inputs.values():
			if (not isinstance(iv, list)) or (not iv) or (iv[0] in to_del):
				continue
			
			recursive_node_deletion(iv[0])

	def find_node(class_type):
		for i in prompt.keys():
			if prompt[i]["class_type"] == class_type:
				return i

	output_src = None
	for i in prompt.keys():
		if prompt[i]["class_type"].startswith("RemoteQueue"):
			if clean_url(prompt[i]["inputs"]["remote_url"]) == remote_url:
				prompt[i]["inputs"]["enabled"] = "remote"
				output_src = i
			else:
				prompt[i]["inputs"]["enabled"] = "false"

	banned = [] if outputs == "any" else ["PreviewImage", "SaveImage"] # get_output_nodes(remote_url)
	output = None
	for i in prompt.keys():
		# only leave current fetch but replace with PreviewImage
		if prompt[i]["class_type"] == "FetchRemote":
			recursive_node_deletion(i)
			if prompt[i]["inputs"]["remote_info"][0] == output_src:
				output = {
					"inputs": {"images": [str(find_node("VAEDecode")), 0]},
					"class_type": 'PreviewImage',
					"final_output": True, # might allow multiple outputs with an ID?
				}

		# do not save output on remote
		if prompt[i]["class_type"] in banned:
			to_del.append(i)
	if output:
		prompt[str(max([int(x) for x in prompt.keys()])+1)] = output
	for i in to_del: del prompt[i]

	### OS LOGIC ###
	sep_remote = "\\" if get_remote_os(remote_url) == "nt" else "/"
	sep_local  = "\\" if os.name == "nt" else "/"
	sem_input_map = { # class type : input to replace
		"CheckpointLoaderSimple" : "ckpt_name",
		"CheckpointLoader"       : "ckpt_name",
		"LoraLoader"             : "lora_name",
		"VAELoader"              : "vae_name",
	}
	if sep_remote != sep_local:
		for i in prompt.keys():
			if prompt[i]["class_type"] in sem_input_map.keys():
				key = sem_input_map[prompt[i]["class_type"]]
				prompt[i]["inputs"][key] = prompt[i]["inputs"][key].replace(sep_local, sep_remote)

	### SEND REQUEST ###
	data = {
		"prompt": prompt,
		"client_id": get_client_id(),
		"extra_data": {
			"job_id": job_id,
		}
	}
	ar = requests.post(
		f"{remote_url}/prompt",
		data    = json.dumps(data),
		headers = {"Content-Type": "application/json"},
		timeout = 4,
	)
	ar.raise_for_status()
	return
