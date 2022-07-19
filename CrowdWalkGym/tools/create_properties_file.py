import os
import glob
import json


def do(path_to_crowdwalk_config_dir, path_to_gym, path_to_run_dir, n_obj):

    with open(path_to_crowdwalk_config_dir + "properties.json", "r", encoding="utf8") as f:
        prop = json.load(f, strict=False)

    prop["map_file"] = prop["map_file"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    prop["generation_file"] = prop["generation_file"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    prop["scenario_file"] = prop["scenario_file"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    prop["fallback_file"] = prop["fallback_file"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    prop["node_appearance_file"] = prop["node_appearance_file"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    prop["link_appearance_file"] = prop["link_appearance_file"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    prop["individual_pedestrians_log_dir"] = prop["individual_pedestrians_log_dir"].replace("path_to_run_dir", path_to_run_dir)

    prop["ruby_load_path"] = prop["ruby_load_path"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    prop["ruby_init_script"] = prop["ruby_init_script"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    prop["ruby_init_script"] = prop["ruby_init_script"].replace("path_to_gym_", path_to_gym)
    prop["ruby_init_script"] = prop["ruby_init_script"].replace("path_to_run_dir", path_to_run_dir)
    prop["ruby_init_script"] = prop["ruby_init_script"].replace("n_obj_", str(n_obj))

    if "polygon_appearance_file" in prop:
        prop["polygon_appearance_file"] = prop["polygon_appearance_file"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)
    if "camera_2d_file" in prop:
        prop["camera_2d_file"] = prop["camera_2d_file"].replace("path_to_crowdwalk_config_dir/", path_to_crowdwalk_config_dir)

    with open(path_to_run_dir + "/properties.json", "w") as f:
        json.dump(prop, f, ensure_ascii=False, indent=4)