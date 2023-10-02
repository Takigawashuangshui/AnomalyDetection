"""simulation of the anomaly detection pipeline
"""
import argparse
import os
from detect_one_image import load_image, perform, save_map

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])                        
    parser.add_argument('-o', '--output_dir', default='output/3')                        
    parser.add_argument('-p', '--sample_path', default='./mvtec_anomaly_detection/metal_nut/test/scratch/000.png') 
    return parser.parse_args()

def main():
    config = get_argparse()
    path = os.path.join(config.sample_path)

    #--- 1: Get image from Image Acquisition Module ---
    image = load_image(path)
    print("load image complete.")

    #--- 2: Perform anomaly detection ---
    map = perform(image=image, path=path, 
                output_dir=config.output_dir, model_size=config.model_size)
    print("Anomaly detection complete.")      

    #--- 3: Save anomaly map ---
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps_example',
                'mvtec_ad')
    save_map(path, test_output_dir, map)
    print("Detection complete. Result can be found in ",test_output_dir)

    #--- 4: Feed the result into the production line for further analysis ---
    print("Send result into production line.")

    #--- 5: Action from production line ---
    print("Remove damaged parts/ ...")


    print("---------------------------")
    print("Anomaly detection pipeline complete.")

if __name__ == '__main__':
    main()