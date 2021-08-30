  
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:43:34 2021
@author: aparravi
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import shutil

##############################
##############################

DEFAULT_REGION = "rNHZ:US-SANJOSE-1-AD-1"
DEFAULT_VM = "VM.Standard.E2.1"
DEFAULT_DEBUG = False
DEFAULT_NUM_GPUS = 0
DEFAULT_PUBLIC_IP = "152.67.254.100"

# Map GPU number to default instance shapes;
NUM_GPU_TO_SHAPE = {
    0: DEFAULT_VM,
    1: "VM.GPU3.1",
    2: "VM.GPU3.2",
    4: "VM.GPU3.4",
    8: "BM.GPU3.8",
}

SSH_KEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDkAFwfA9p+fEiJMECYGtBFde13EVeCeEUawMaYkzQWdRP3LEwh+D4mqJtKAGJImn0kayYjT/nJg+JHKycZh114GaRvW+VyinjrdXrQSZJC1wcN9+uy3U2V2qSRMthHZxs+xHeAzBZSAQzcCaSuq64XojPfsLzXct0n72Ej4CGeTjo33J5ak0IqCs9qwhIsvm4241c3gO0e17L23EE9sG8lzh+m8FpJyeon+QkLg7yNqhrsL5lqUomSXFZTswvg6J1cFotoa57EzQ44z4uEHG3kMb/Bg4HRCLT4jBwmFmzKQn2R+rkUoC0KxtGuPCrhxjxq7jGJXlg5fN0qMWZRmF0H aparravi@DESKTOP-L90IVGL"

DEFAULT_SETUP_JSON = """
{{
  "compartmentId": "ocid1.compartment.oc1..aaaaaaaakrrqfga2bqdbv7ruydhh36cbyy46w2m6k4dnbdfw3gyes45qebfq",
  "sourceBootVolumeId": "ocid1.bootvolume.oc1.us-sanjose-1.abzwuljrtlv5wmipcguqvbgo5s7ctuhrw4v2pjpbat4iggfm4dpzcjcoonwq",
  "sshAuthorizedKeys": "{}",
  "subnetId": "ocid1.subnet.oc1.us-sanjose-1.aaaaaaaapwj5a3hkbym7yyeqha6chagx23lifuxnwxayfporkhqw2zuemdoq",
  "assignPublicIp": false
}}
""".format(SSH_KEY)

# Temporary directory where data are stored;
DEFAULT_TEMP_DIR = "tmp_oci_setup"

# OCI commands;
OCI_LAUNCH_INSTANCE = "oci compute instance launch --from-json file://{} --wait-for-state RUNNING"
OCI_OBTAIN_VNIC = "oci compute instance list-vnics --limit 1 --instance-id {}"
OCI_OBTAIN_PRIVATE_IP = "oci network private-ip list --vnic-id {}"
OCI_OBTAIN_PUBLIC_IP = "oci network public-ip get --public-ip-address {}"
OCI_UPDATE_PUBLIC_IP = "oci network public-ip update --public-ip-id {} --private-ip-id {}"

##############################
##############################

def log_message(message: str) -> None:
    date = datetime.now()
    date_str = date.strftime("%Y-%m-%d-%H-%M-%S-%f")
    print(f"[{date_str} oci-setup] {message}")


def parse_shape_name(shape: str) -> str:
    if shape == DEFAULT_VM:
        return "cpu-default"
    elif "gpu" in shape.lower():
        gpu_count = shape.split(".")[-1]
        return f"gpu-{gpu_count}"
    else:
        return shape.replace(".", "-")


def create_instance_launch_dict(shape: str, debug: bool=DEFAULT_DEBUG) -> dict:
    instance_launch_dict = json.loads(DEFAULT_SETUP_JSON)
    # Add region;
    instance_launch_dict["availabilityDomain"] = DEFAULT_REGION
    # Add shape;
    instance_launch_dict["shape"] = shape
    # Create hostname and display name;
    hostname = display_name = parse_shape_name(shape)
    instance_launch_dict["hostname"] = f"grcuda-{hostname}"
    instance_launch_dict["displayName"] = f"grcuda-{display_name}"
    if debug:
        log_message(instance_launch_dict)
    return instance_launch_dict


def run_oci_command(command_template: str, *command_format_args, debug: bool=DEFAULT_DEBUG, ) -> dict:
    # Setup the OCI command;
    oci_command = command_template.format(*command_format_args)
    if debug:
        log_message(f"launching OCI command: {oci_command}")
    # Launch the OCI command;
    try:
        result = subprocess.run(oci_command, shell=True, env=os.environ, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        if debug:
            log_message(f"caught exception {e.output} during OCI command")
        exit(-1)
    if result.stderr:
        if debug:
            log_message("OCI command completed with error")
            log_message(result.stderr)
        exit(-1)
    # Everything is ok, we extract the result as a dictionary.
    # There might be other stuff printed along with the JSON, so we have to remove it;
    res_tmp = result.stdout.decode("utf-8")
    res_tmp = res_tmp[res_tmp.index("{"):]  # Delete everything until the first "{";
    res_tmp = res_tmp[:-res_tmp[::-1].index("}")]  # Delete everything after the last "}";
    return json.loads(res_tmp)


def launch_instance(instance_launch_dict: dict, debug: bool=DEFAULT_DEBUG) -> str:
    # We have to store the dictionary to a temporary file;
    launch_json_file_name = os.path.join(DEFAULT_TEMP_DIR, instance_launch_dict["displayName"] + ".json") 
    if debug:
        log_message(f"storing temporary launch JSON into {launch_json_file_name}")
    # Create temporary folder;
    Path(DEFAULT_TEMP_DIR).mkdir(parents=True, exist_ok=True)
    # Store dictionary to JSON;
    with open(launch_json_file_name, "w") as f:
        json.dump(instance_launch_dict, f)
    # Setup the launch command;
    result = run_oci_command(OCI_LAUNCH_INSTANCE, launch_json_file_name, debug=debug)
    # Extract the instance OCID for later use;
    instance_ocid = result["data"]["id"]
    if debug:
        log_message(f"created instance with OCID={instance_ocid}")

    # Remove the temporary configuration file;
    os.remove(launch_json_file_name)
    if len(os.listdir(DEFAULT_TEMP_DIR)) == 0: 
        shutil.rmtree(DEFAULT_TEMP_DIR)  # Remove the folder if it is empty;

    return instance_ocid


def attach_reserved_public_ip(instance_ocid: str, debug: bool=DEFAULT_DEBUG) -> None:
    # We have to obtain the VNIC attached to the instance (assume only 1 VNIC is available);
    result = run_oci_command(OCI_OBTAIN_VNIC, instance_ocid, debug=debug)
    # Extract the VNIC OCID;
    vnic_ocid = result["data"][0]["id"]
    if debug:
        log_message(f"obtained VNIC with OCID={vnic_ocid}")
    # Obtain the private address OCID associated to the VNIC;
    result = run_oci_command(OCI_OBTAIN_PRIVATE_IP, vnic_ocid, debug=debug)
    # Extract the private IP OCID;
    private_ip_ocid = result["data"][0]["id"]
    if debug:
        log_message(f"obtained private IP with OCID={private_ip_ocid}")
    # Obtain the public IP OCID;
    result = run_oci_command(OCI_OBTAIN_PUBLIC_IP, DEFAULT_PUBLIC_IP, debug=debug)
    # Extract the VNIC OCID;
    public_ip_ocid = result["data"]["id"]
    if debug:
        log_message(f"obtained public IP with OCID={public_ip_ocid}")
    # Assign the reserved public IP;
    run_oci_command(OCI_UPDATE_PUBLIC_IP, public_ip_ocid, private_ip_ocid, debug=debug)
    if debug:
        log_message(f"assigned public IP {DEFAULT_PUBLIC_IP}")

##############################
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup OCI instances from the command line")
    parser.add_argument("-d", "--debug", action="store_true", help="If present, print debug messages", default=DEFAULT_DEBUG)
    parser.add_argument("-g", "--num_gpus", metavar="N", type=int, default=DEFAULT_NUM_GPUS, help="Number of GPUs present in the instance")
   
    # 1. Parse the input arguments;
    args = parser.parse_args()
    debug = args.debug
    num_gpus = args.num_gpus

    # 2. Select shape;
    shape = DEFAULT_VM
    if num_gpus in NUM_GPU_TO_SHAPE:
        shape = NUM_GPU_TO_SHAPE[num_gpus]
    if debug:
        log_message(f"using {num_gpus} GPUs")
        log_message(f"selected shape {shape}")

    # 3. Obtain configuration dictionary;
    instance_launch_dict = create_instance_launch_dict(shape, debug)

    # 4. Launch the instance;
    instance_id = launch_instance(instance_launch_dict, debug)

    # 5. Attach the reserved public IP to the instance;
    attach_reserved_public_ip(instance_id, debug)

    if debug:
        log_message("setup completed successfully!")