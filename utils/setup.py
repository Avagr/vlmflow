from pathlib import Path

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, GenerationConfig

from datasets.base import EvalWrapper
from datasets.gqa import GQA, GQAEval, GQACollate
from datasets.coco import UnlabeledCoco, UnlabeledCocoCollate, UnlabeledCocoEval
from datasets.mmvp import MMVP, MMVPCollate, MMVPEval
from datasets.seedbench import SEEDBenchSingleImage, SEEDBenchSingleImageEval, SEEDBenchCollate
from datasets.whatsup import WhatsUp, WhatsUpEval, WhatsUpCollate
from flow.analysis import *
from models.transparent_models import TransparentLlava
from models.wrappers import GenerativeWrapper
from utils.callbacks import GraphTensorCallback, LogitLensCallback


def create_model(cfg, device):
    match cfg.model.dtype:
        case "bfloat16":
            dtype = torch.bfloat16
        case "float16":
            dtype = torch.float16
        case "float32":
            dtype = torch.float32
        case _:
            raise ValueError(f"Unsupported dtype '{cfg.model.dtype}'")

    match cfg.model.name:
        case "llava":

            llava = LlavaForConditionalGeneration.from_pretrained(cfg.model.config_name,
                                                                  torch_dtype=dtype,
                                                                  low_cpu_mem_usage=True,
                                                                  attn_implementation=cfg.model.attn_impl)

            processor = AutoProcessor.from_pretrained(cfg.model.processor_path)
            model = TransparentLlava(cfg.model.name, llava, processor, device, dtype)
            model = GenerativeWrapper(processor, model, device, dtype, output_attentions=True)
        case _:
            raise ValueError(f"Unsupported model '{cfg.model.name}'")

    return model.to(device)


def create_eval_task(cfg, device):
    sampling_config = GenerationConfig(**cfg.sampling_params)
    match cfg.task.name:
        case "SEED-Bench-2":
            dataset = SEEDBenchSingleImage(cfg.task.task_num, Path(cfg.task.json_path), Path(cfg.task.image_root))
            collate = SEEDBenchCollate()
            wrapper = SEEDBenchSingleImageEval(cfg.prompt.text, device, cfg.task.eval_method)

        case "WhatsUp":
            if cfg.task.part == 'A':
                json_path = Path(cfg.task.json_path_A)
            elif cfg.task.part == 'B':
                json_path = Path(cfg.task.json_path_B)
            else:
                raise ValueError(f"Unsupported What's Up part '{cfg.task.part}'")
            dataset = WhatsUp(Path(cfg.task.image_root), json_path, permute_options=cfg.task.permute)
            collate = WhatsUpCollate()
            wrapper = WhatsUpEval(cfg.prompt.text, device, cfg.task.eval_method)

        case "GQA":
            dataset = GQA(Path(cfg.task.test_question_file), Path(cfg.task.img_dir))
            collate = GQACollate()
            wrapper = GQAEval(cfg.prompt.text, device, cfg.task.eval_method, sampling_config)

        case "UnlabeledCOCO":
            dataset = UnlabeledCoco(Path(cfg.task.img_dir), Path(cfg.task.img_descriptions_file), cfg.task.dataset_size)
            collate = UnlabeledCocoCollate()
            wrapper = UnlabeledCocoEval(cfg.prompt.text, device, sampling_config)

        case "MMVP":
            dataset = MMVP(Path(cfg.task.csv_path), Path(cfg.task.img_dir))
            collate = MMVPCollate()
            wrapper = MMVPEval(cfg.prompt.text, cfg.task.eval_method)

        case _:
            raise ValueError(f"Unsupported task '{cfg.task.name}'")

    return dataset, wrapper, collate


def create_callbacks(cfg, eval_wrapper: EvalWrapper, run_folder: Path, model: GenerativeWrapper):
    if cfg.save_full_graphs:
        (run_folder / "graphs").mkdir(parents=True, exist_ok=True)
        # callback = GraphCallback(run_folder / "graphs", cfg.renormalization_threshold, cfg.sparsification_threshold)
        callback = GraphTensorCallback(run_folder / "graphs", head_batch_size=cfg.head_batch_size)
        eval_wrapper.add_callback(callback)
    if cfg.last_token_logit_lens:
        callback = LogitLensCallback(["A", "B", "C", "D"], normalize_lens=cfg.normalize_lens,
                                     tokenizer=model.processor.tokenizer)
        eval_wrapper.add_callback(callback)
    return eval_wrapper.callbacks


def create_metrics(cfg) -> tuple[list[BaseVertexMetric], list[BaseGraphMetric]]:
    vertex_metrics = []
    if cfg.modality_contribution:
        vertex_metrics.append(ModalityContribution())
    if cfg.modality_centrality:
        vertex_metrics.append(ModalityCentrality())
    if cfg.local_clustering_coefficient:
        vertex_metrics.append(LocalClusteringCoefficient())

    graph_metrics = []
    if cfg.graph_density:
        graph_metrics.append(GraphDensity())
    if cfg.num_cross_modal_edges:
        graph_metrics.append(NumCrossModalEdges())
    if cfg.global_clustering_coefficient:
        graph_metrics.append(GlobalClusteringCoefficient())

    return vertex_metrics, graph_metrics