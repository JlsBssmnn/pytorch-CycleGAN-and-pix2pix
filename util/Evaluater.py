import warnings
from models.custom_transforms import Scaler
import numpy as np
import torch
import h5py
import sys
import pathlib
from collections import defaultdict
from util.logging_config import logging

class EpithelialEvaluater:
    def __init__(self, config):
        sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
        global apply_generator, Evaluater
        from evaluation.translate_image import apply_generator
        from evaluation.evaluate_image import Evaluater

        self.config = config.evaluation_config

        self.membrane_truths = []
        self.cell_truths = []
        with h5py.File(self.config.ground_truth_file) as f:
            for datasets in self.config.ground_truth_datasets:
                self.membrane_truths.append(np.asarray(f[datasets[0]]))
                self.cell_truths.append(np.asarray(f[datasets[1]]))

        self.images = []
        transform = Scaler(config.input_value_range[0], config.input_value_range[1], -1, 1)
        with h5py.File(self.config.input_file) as f:
            for s in self.config.image_slices:
                image = np.asarray(eval(f"f[self.config.input_dataset][{s}]"))
                image = torch.tensor(image)
                image = transform(image)
                self.images.append(image)
        self.image_names = config.evaluation_config.image_names
        self.evaluater = Evaluater(self.config, False)
        self.net_out_transform = Scaler(config.generator_output_range[0], config.generator_output_range[1], 0, 1)

        logging.info("Epithelial evaluater created")

    def compute_evaluation(self, generator):
        outputs = []
        for image in self.images:
            output = apply_generator(image, generator, self.config)
            output = self.net_out_transform(torch.from_numpy(output)).numpy()
            outputs.append(output)
        self.evaluater.clear()
        self.evaluater.find_segmentation_and_eval(outputs)

        scores = defaultdict(lambda: [])
        for evaluation in self.evaluater.results["evaluation"]:
            tweak_image = evaluation["segmentation_parameters"]["tweak_image"]
            eval_scores = evaluation["evaluation_scores"]
            for image_name in [x for x in self.image_names if x != tweak_image]:
                if "variation_of_information" in eval_scores[image_name]:
                    scores[image_name + "_VI"].append(eval_scores[image_name]["variation_of_information"])
                if "score" in eval_scores[image_name]:
                    scores[image_name + "_score"].append(eval_scores[image_name]["score"])
            scores[tweak_image + "_diff"].append(eval_scores[tweak_image]["diff"])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            for image_name, score_values in scores.items():
                scores[image_name] = np.nanmean(score_values)
        return dict(sorted(scores.items()))
