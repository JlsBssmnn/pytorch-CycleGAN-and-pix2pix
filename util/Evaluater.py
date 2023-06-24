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
        from evaluation.translate_image import GeneratorApplier
        from evaluation.evaluate_epithelial import SEEpithelial

        self.config = config.evaluation_config

        self.membrane_truths = []
        self.cell_truths = []
        with h5py.File(self.config.ground_truth_file) as f:
            for datasets in self.config.ground_truth_datasets:
                self.membrane_truths.append(np.asarray(f[datasets[0]]))
                self.cell_truths.append(np.asarray(f[datasets[1]]))

        self.images = []
        image_shape = None
        transform = Scaler(config.input_value_range[0], config.input_value_range[1], -1, 1)
        with h5py.File(self.config.input_file) as f:
            for s in self.config.image_slices:
                image = np.asarray(eval(f"f[self.config.input_dataset][{s}]"))
                image = torch.tensor(image)
                image = transform(image)
                if self.config.use_gpu:
                    image = image.cuda(config.gpu_ids[0])

                if image_shape is None:
                    image_shape = image.shape
                else:
                    assert image_shape == image.shape, "All evaluation image must have the same shape"

                self.images.append(image)

        self.image_names = config.evaluation_config.image_names
        self.generator_applier = GeneratorApplier(self.images[0].shape, self.config)
        self.evaluater = SEEpithelial(self.config, False)
        self.net_out_transform = Scaler(config.generator_output_range[0], config.generator_output_range[1], 0, 1)

        logging.info("Epithelial evaluater created")

    def compute_evaluation(self, generator, total_iters):
        if total_iters % self.config.eval_freq != 0:
            return {}
        elif total_iters % self.config.vi_freq != 0:
            return self.compute_evaluation_without_VI(generator)
        else:
            return self.compute_evaluation_with_VI(generator)

    def compute_evaluation_without_VI(self, generator):
        scores = {}
        generator.eval()

        for i, image in enumerate(self.images):
            output = self.generator_applier.apply_generator(image, generator)
            output = self.net_out_transform(torch.from_numpy(output)).numpy()
            output = eval(f'output[{self.config.slice_str}]')
            diff = self.evaluater.compute_diff(output, i)
            scores[self.image_names[i] + "_diff"] = diff

        generator.train()
        return scores

    def compute_evaluation_with_VI(self, generator):
        outputs = []
        generator.eval()
        for image in self.images:
            output = self.generator_applier.apply_generator(image, generator)
            output = self.net_out_transform(torch.from_numpy(output)).numpy()
            outputs.append(output)
        generator.train()
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


class BrainbowEvaluater:
    def __init__(self, config):
        sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
        global apply_generator, Evaluater
        from evaluation.translate_image import GeneratorApplier
        from evaluation.evaluate_brainbows import SEBrainbow

        self.config = config.evaluation_config

        self.images = []
        image_shape = None
        with h5py.File(self.config.input_file) as f:
            entire_image = np.asarray(f[self.config.input_dataset])
            transform = Scaler(0, entire_image.max(), -1, 1)

            for s in self.config.image_slices:
                image = eval(f"entire_image[{s}]")
                if image.dtype == np.uint16:
                    image = image.astype(np.float32)
                image = torch.tensor(image)
                if not self.config.scale_with_patch_max:
                    image = transform(image)
                if self.config.use_gpu:
                    image = image.cuda(config.gpu_ids[0])

                if image_shape is None:
                    image_shape = image.shape
                else:
                    assert image_shape == image.shape, \
                    f"All evaluation images must have the same shape, but {image_shape} != {image.shape}"

                self.images.append(image)

        self.image_names = config.evaluation_config.image_names
        self.generator_applier = GeneratorApplier(self.images[0].shape, self.config)
        self.evaluater = SEBrainbow(self.config)
        self.net_out_transform = Scaler(config.generator_output_range[0], config.generator_output_range[1], 0, 1)

        logging.info("Brainbow evaluater created")

    def compute_evaluation(self, generator, total_iters):
        if total_iters % self.config.eval_freq != 0:
            return {}
        
        outputs = []
        generator.eval()
        for image in self.images:
            output = self.generator_applier.apply_generator(image, generator)
            output = self.net_out_transform(torch.from_numpy(output)).numpy()
            outputs.append(output)
        generator.train()
        self.evaluater.clear()
        self.evaluater.find_segmentation_and_eval(outputs, total_iters % self.config.vi_freq == 0)

        scores = defaultdict(lambda: [])
        for evaluation in self.evaluater.results["evaluation"]:
            tweak_image = evaluation["segmentation_parameters"]["tweak_image"]
            eval_scores = evaluation["evaluation_scores"]
            for image_name in [x for x in self.image_names if x != tweak_image]:
                if "variation_of_information" in eval_scores[image_name]:
                    scores[image_name + "_VI"].append(eval_scores[image_name]["variation_of_information"])
                    scores[image_name + "_under_seg"].append(eval_scores[image_name]["under_segmentation"])
                    scores[image_name + "_over_seg"].append(eval_scores[image_name]["over_segmentation"])
            scores[tweak_image + "_fg_prec"].append(eval_scores[tweak_image]["foreground_prec"])
            scores[tweak_image + "_fg_rec"].append(eval_scores[tweak_image]["foreground_rec"])
            scores[tweak_image + "_fg_f1"].append(eval_scores[tweak_image]["foreground_f1"])
            scores[tweak_image + "_fg_acc"].append(eval_scores[tweak_image]["foreground_acc"])
            scores[tweak_image + "_fg_diff"].append(eval_scores[tweak_image]["foreground_diff"])
            scores[tweak_image + "_affinity_diff"].append(eval_scores[tweak_image]["affinity_diff"])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            for image_name, score_values in scores.items():
                scores[image_name] = np.nanmean(score_values)
        return dict(sorted(scores.items()))
