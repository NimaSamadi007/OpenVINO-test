import os
import numpy as np
import cv2 as cv
import argparse

from openvino.tools.pot import DataLoader   
from openvino.tools.pot import IEEngine
from openvino.tools.pot import load_model, save_model
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline 

class ImageLoader(DataLoader):
    def __init__(self, dataset_path):
        self._files = []
        all_files_in_dir = os.listdir(dataset_path)
        for file_name in all_files_in_dir:
            file = os.path.join(dataset_path, file_name)
            if cv.haveImageReader(file): # Check if file is an image
                self._files.append(file)

    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index):
        if index > len(self):
            raise IndexError("Index out of range")
        # preprocessing is also done in this function
        img = cv.imread(self._files[index], cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = applyLetterBox(img)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, 3, img.shape[0], img.shape[1])
        return (img, None)

def applyLetterBox(img):
    res_img = img
    r = min(1.0*640/img.shape[0], 1.0*640/img.shape[0])
    new_unpad = np.array([round(1.0 * img.shape[1] * r),
                          round(1.0 * img.shape[0] * r)], dtype=int)
    dw = ((640 - new_unpad[0]) % 32) / 2.0
    dh = ((640 - new_unpad[1]) % 32) / 2.0

    if img.shape[1] != new_unpad[0] and img.shape[0] != new_unpad[1]:
        res_img = cv.resize(img, (new_unpad[0], new_unpad[1]))
    
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    res_img = cv.copyMakeBorder(res_img, top, bottom, left, right,
                                cv.BORDER_CONSTANT, value=[114, 114, 114])
    return res_img

def main():
    parser = argparse.ArgumentParser(description='Fire and Smoke Classification Model Quantizer')
    parser.add_argument('-m', '--model-name', type=str, dest='model_name', help="FP32 model name that will be quantized", default='b12')
    parser.add_argument('-d', '--dataset-path', type=str, required=True, dest='dataset_path', help="Dataset path that contains validation data")
    parser.add_argument('-s', '--stat-subset-size', type=int, dest='stat_subset_size', help="stat-subset-size that will be used in quantization", default=300)
    parser.add_argument('-v', '--verbose', help="verbose messages", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        from openvino.tools.pot.utils import logger
        logger.init_logger("DEBUG")

    model_name = args.model_name
    model_config = {
        "model_name": model_name,
        "model": "./../models/fscls/" + model_name + "/" + model_name + ".xml",
        "weights": "./../models/fscls/" + model_name + "/" + model_name + ".bin",
    }

    engine_config = {
        "device" : "CPU", 
        "type": "simplified", 
        "stat_requests_number": 1, 
        "eval_requests_number": 1
    }

    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "CPU",
                "stat_subset_size": args.stat_subset_size,
            },
        }
    ]

    data_loader = ImageLoader(args.dataset_path)

    model = load_model(model_config=model_config)
    print("Model loaded")

    engine = IEEngine(config=engine_config, data_loader=data_loader)
    print("Engine created")

    pipeline = create_pipeline(algorithms, engine)
    print("Pipeline created")
    print("Running pipeline")
    compressed_model = pipeline.run(model=model)
    print("pipeline finished...")

    print("compressing model weights....")
    compress_model_weights(compressed_model)

    print("saving quantized model...")
    compressed_model_paths = save_model(
        model = compressed_model,
        save_path="./../models/fscls/" + model_name + "-quan/",
        model_name=model_name + "-quan"
    )

if __name__ == "__main__":
    main()
