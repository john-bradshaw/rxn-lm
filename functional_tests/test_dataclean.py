
import filecmp
import os
from os import path as osp
import shutil
import subprocess
import uuid


def test_dataclean_procedure():
    this_dir_name = osp.dirname(__file__)
    os.chdir(this_dir_name)

    # Setup an experiment folder
    exp_name = str(uuid.uuid4())
    experiment_folder = osp.join("test_op", exp_name)
    os.makedirs(experiment_folder, exist_ok=False)

    # Run the clean data
    params = {
        "input_file": osp.join(this_dir_name, "dataclean_test_files", "dataclean_test_input.txt"),
        "output_file": osp.join(this_dir_name, experiment_folder, "01_datacleaned_output.txt"),
        "remove_atmmap": True,
        "canonicalize": True,
        "remove_on_error": True
    }
    os.chdir(osp.join(this_dir_name, "../scripts", "data_prep"))
    subprocess.run(["python", "01_clean_data.py", "--input_file", params["input_file"], "--output_file", params["output_file"], "--remove_atmmap", str(params["remove_atmmap"]), "--canonicalize", str(params["canonicalize"]), "--remove_on_error", str(params["remove_on_error"])], check=True)

    # Run the vocab creator
    params = {
        "input_file": osp.join(this_dir_name, experiment_folder, "01_datacleaned_output.txt"),
        "vocab_fname": osp.join(this_dir_name, experiment_folder, "02_vocab_output.json"),
    }
    subprocess.run(["python", "02_create_vocab.py", "--files_to_read", params["input_file"], "--vocab_fname", params["vocab_fname"]], check=True)

    # Run the jsonl creator
    params = {
        "input_file": osp.join(this_dir_name, experiment_folder, "01_datacleaned_output.txt"),
        "shuffle": True,
        "random_seed": 42
    }
    subprocess.run(["python", "03_convert_to_jsonl.py", "--input_file", params["input_file"], "--shuffle", str(params["shuffle"]), "--random_seed", str(params["random_seed"])], check=True)

    assert filecmp.cmp(osp.join(this_dir_name, "dataclean_test_files/expected_01_datacleaned_output.txt"), osp.join(this_dir_name, experiment_folder, "01_datacleaned_output.txt"), shallow=False)
    assert filecmp.cmp(osp.join(this_dir_name, "dataclean_test_files/expected_01_datacleaned_output.jsonl"), osp.join(this_dir_name, experiment_folder, "01_datacleaned_output.jsonl"), shallow=False)
    assert filecmp.cmp(osp.join(this_dir_name, "dataclean_test_files/expected_02_vocab_output.json"), osp.join(this_dir_name, experiment_folder, "02_vocab_output.json"), shallow=False)

    shutil.rmtree(osp.join(this_dir_name, experiment_folder))

