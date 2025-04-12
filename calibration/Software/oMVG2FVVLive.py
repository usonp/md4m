import os
import json
import yaml
import sys
import numpy as np

def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.executable} {sys.argv[0]} <input OpenMVG JSON> <detected depth folder> <num servers> <output FVVLive YAML>")
        sys.exit(-1)

    input_omvg = sys.argv[1]
    detected_depth = sys.argv[2]
    num_servers = int(sys.argv[3])
    output_fvv = sys.argv[4]


    # Select the cameras present in detected depth folder to use them for syhtesisConfiguration
    SC_view_list = []
    for filename in os.listdir(detected_depth):
        if filename.endswith(".txt"):
            id = int(filename.split('.')[0][-2:])
            SC_view_list.append(id)

    print(f'{len(SC_view_list)} cameras selected: {SC_view_list}')

    transform_openmvg_to_fvv(input_omvg, output_fvv, SC_view_list, num_servers=num_servers)

# main


def select_views(sfm_data, view_list):
    return [view for view in sfm_data["views"] if view["key"] in view_list]


def transform_openmvg_to_fvv(input_json, output_fvv, view_list, num_servers):
    with open(input_json, 'r') as f:
        sfm_data = json.load(f)

    selected_views = select_views(sfm_data, view_list)
    num_cameras = len(selected_views)

    # Print selected views (weird way to do it in one line)
    print("\n".join("Camera {} - {}".format(i, view["value"]["ptr_wrapper"]["data"]["filename"]) for i, view in enumerate(selected_views)))

    # Work
    result = {"IntrinsicAndExtrinsic": []}

    for i, view in enumerate(selected_views):
        view_id = view["key"]
        pose_id = view["value"]["ptr_wrapper"]["data"]["id_pose"]
        intrinsic_id = view["value"]["ptr_wrapper"]["data"]["id_intrinsic"]

        intrinsic = sfm_data["intrinsics"][intrinsic_id]["value"]["ptr_wrapper"]["data"]
        intrinsic_matrix = [
            [intrinsic["focal_length"], 0.0, intrinsic["principal_point"][0]],
            [0.0, intrinsic["focal_length"], intrinsic["principal_point"][1]],
            [0.0, 0.0, 1.0]
        ]


        extrinsic = sfm_data["extrinsics"][pose_id]["value"]
        rotation = extrinsic["rotation"]
        center = extrinsic["center"]

        R = np.array(rotation)
        c = np.array(center)
        t = (-R) @ c 

        extrinsic_matrix = [
            [rotation[0][0], rotation[0][1], rotation[0][2], float(t[0])],
            [rotation[1][0], rotation[1][1], rotation[1][2], float(t[1])],
            [rotation[2][0], rotation[2][1], rotation[2][2], float(t[2])]
        ]

        # Determine Server and Streams_Camera based on view_id
        cams_per_server = round(num_cameras/num_servers)
        server = i // cams_per_server
        streams_camera = i % cams_per_server

        result["IntrinsicAndExtrinsic"].append({
            "CameraId": i,
            "OriginalId": view_id,
            "Server": server,
            "Streams_Camera": streams_camera,
            "intrinsic": intrinsic_matrix,
            "extrinsic": extrinsic_matrix
        })

    with open(output_fvv, 'w') as f:
        yaml.dump(result, f, indent=3, default_flow_style=True, sort_keys=False)



if __name__ == "__main__":
    main()
