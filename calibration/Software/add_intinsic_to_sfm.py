import sys
import json

def main():
    
    if len(sys.argv) != 4:
        print(f"Usage: {sys.executable} {sys.argv[0]} <OpenMVG JSON> <Intrinsic JSON> <Use Rectified Intrinsics>")
        sys.exit(-1)

    input_omvg = sys.argv[1]
    input_intrinsic = sys.argv[2]
    rectified = bool(int(sys.argv[3]))

    sfm__data = load_json(input_omvg)
    intrinsics_data = load_json(input_intrinsic)

    # Select original intrinsics or rectified intrinsics
    if rectified:
        camera_intrinsics = 'new_camera_matrix'
        print(f'Using rectified camera intrinsics, {camera_intrinsics}')
    else:
        camera_intrinsics = 'camera_matrix'
        print(f'Using original camera intrinsics, {camera_intrinsics}')

    for view in sfm__data['views']:

        id = view['key']

        for intrinsic in sfm__data['intrinsics']:
            if intrinsic['key'] == view['value']['ptr_wrapper']['data']['id_intrinsic']:
                
                # Extract the camera intrinsics from your data
                fx = intrinsics_data[str(id)][camera_intrinsics][0][0]
                fy = intrinsics_data[str(id)][camera_intrinsics][1][1]
                cx = intrinsics_data[str(id)][camera_intrinsics][0][2]
                cy = intrinsics_data[str(id)][camera_intrinsics][1][2]

                print(f'Camera {id} - fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}')

                intrinsic['value']['ptr_wrapper']['data']['focal_length'] = fx
                intrinsic['value']['ptr_wrapper']['data']['principal_point'] = [cx, cy]
                break

        # Save the updated OpenMVG camera file
        save_json(sfm__data, input_omvg)
    


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save JSON data to file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    

if __name__ == '__main__':
    main()