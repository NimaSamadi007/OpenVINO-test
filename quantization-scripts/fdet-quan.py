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
        self.target_width = 320
        self.target_height = 320    
        self.mean_value = 127.5
        self.std_value = 128.0    
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
        img = self.resizeImg(img)
        img = img.reshape(3, img.shape[0], img.shape[1])
        img = (img - self.mean_value) / (self.std_value)
        img = img.reshape(1, 3, img.shape[1], img.shape[2])
        return (img, None)

    def resizeImg(self, img):
        source_ratio = 1.0 * img.shape[1] / img.shape[0]
        target_ratio = 1.0 * self.target_height / self.target_width
        if source_ratio > target_ratio:
            new_height = self.target_height
            new_width = int(np.round(1.0 * new_height / source_ratio))
        else:
            new_width = self.target_width
            new_height = int(np.round(1.0 * new_width * source_ratio))    

        res_img = np.zeros((self.target_width, self.target_height, 3))
        tmp_img = cv.resize(img, [new_height, new_width], 0, 0, interpolation=cv.INTER_LINEAR)
        res_img[0:new_width, 0:new_height, :] = tmp_img

        return res_img


def main():
    parser = argparse.ArgumentParser(description='Face Detection Model Quantizer')
    parser.add_argument('-m', '--model-name', type=str, dest='model_name', help="FP32 model name that will be quantized", default='scrfd_10g_bnkps')
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
        "model": "./../models/fdet/" + model_name + "/" + model_name + ".xml",
        "weights": "./../models/fdet/" + model_name + "/" + model_name + ".bin",
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
    # step 1:
    data_loader = ImageLoader(args.dataset_path)

    # step 2:
    model = load_model(model_config=model_config)
    print("Model loaded")

    # step 3:
    engine = IEEngine(config=engine_config, data_loader=data_loader)
    print("Engine created")

    # step 4:
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
        save_path="./../models/fdet/" + model_name + "-quan/",
        model_name=model_name + "-quan"
    )

if __name__ == "__main__":
    main()
