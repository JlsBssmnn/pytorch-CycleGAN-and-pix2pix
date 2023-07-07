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

        return summarize_results(self.evaluater.results, self.image_names,
                                 [('variation_of_information', 'VI'), 'score'], ['diff'])


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
        self.n_aff = len(config.evaluation_config.offsets)

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

        return summarize_results(self.evaluater.results, self.image_names,
             [('variation_of_information', 'VI'), ('under_segmentation', 'under_seg'), ('over_segmentation', 'over_seg'),
              ('weighted_VI', 'w_VI'), ('weighted_under_seg', 'w_under_seg'), ('weighted_over_seg', 'w_over_seg'),],
             [('foreground_prec', 'fg_prec'), ('foreground_rec', 'fg_rec'), ('foreground_f1', 'fg_f1'),
              ('foreground_acc', 'fg_acc'), ('foreground_diff', 'fg_diff'), 'affinity_diff',
              'affinity_prec', 'affinity_rec', 'affinity_f1'] + [f'affinity_diff_{i}' for i in range(1, self.n_aff + 1)])

def summarize_results(results, image_names, aggregate_metrics: list[str | tuple[str, str]],
                     non_aggregate_metrics: list[str | tuple[str, str]]):
    """
    Summarizes and aggregates the `results` property from either the SEEpithelial or SEBrainbow class.

    Metrics in the `aggregate_metrics` list are averaged for each slice over all evaluations that that were not tweaked
    on that slice. Metrics can be strings, were the metric will be named the same in the resulting dict or a tuple where
    the first element is the name in the computed evaluation and the second element is the name under which the loss is
    returned to the visualizer.

    Metrics in `non_aggregate_metrics` will be extracted for each slice for the evaluation that was tweaked on that
    slice. The format is the same as for the `aggregate_metrics`.
    """
    scores = defaultdict(lambda: [])
    for evaluation in results['evaluation']:
        tweak_image = evaluation["segmentation_parameters"]["tweak_image"]
        eval_scores = evaluation["evaluation_scores"]
        for image_name in [x for x in image_names if x != tweak_image]:
            for metric in aggregate_metrics:
                source_name = metric if type(metric) == str else metric[0]
                target_name = metric if type(metric) == str else metric[1]
                if source_name in eval_scores[image_name]:
                    scores[f"{image_name}_{target_name}"].append(eval_scores[image_name][source_name])
        for metric in non_aggregate_metrics:
            source_name = metric if type(metric) == str else metric[0]
            target_name = metric if type(metric) == str else metric[1]
            scores[f"{tweak_image}_{target_name}"].append(eval_scores[tweak_image][source_name])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for image_name, score_values in scores.items():
            scores[image_name] = np.nanmean(score_values)
    return dict(sorted(scores.items()))
