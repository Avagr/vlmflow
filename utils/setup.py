from pathlib import Path

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, GenerationConfig

from datasets.base import EvalWrapper
from datasets.coco import UnlabeledCoco, UnlabeledCocoCollate, UnlabeledCocoEval
from datasets.gqa import GQA, GQAEval, GQACollate
from datasets.mmvp import MMVP, MMVPCollate, MMVPEval
from datasets.seedbench import SEEDBenchSingleImage, SEEDBenchSingleImageEval, SEEDBenchCollate
from datasets.whatsup import WhatsUp, WhatsUpEval, WhatsUpCollate
from flow.analysis import *
from models.molmo.modeling_molmo import MolmoForCausalLM
from models.molmo.preprocessing_molmo import MolmoProcessor
from models.transparent_models import TransparentLlava, TransparentMolmo, TransparentPixtral
from models.wrappers import GenerativeWrapper
from utils.callbacks import GraphTensorCallback, LogitLensCallback, ResidualsByLayerCallback, ImageBoundariesCallback


def create_model(cfg):
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
                                                                  device_map='auto',
                                                                  attn_implementation=cfg.model.attn_impl)

            processor = AutoProcessor.from_pretrained(cfg.model.processor_path)
            model = TransparentLlava(cfg.model.name, llava, processor, llava.device, dtype)
            model = GenerativeWrapper(processor, model, llava.device, dtype, list(cfg.model.vqa_tokens),
                                      output_attentions=True)
        case "molmo":
            molmo = MolmoForCausalLM.from_pretrained(
                cfg.model.config_name,
                torch_dtype=dtype,
                device_map='auto',
                low_cpu_mem_usage=True,
            )
            molmo.model.vision_backbone = molmo.model.vision_backbone.to(torch.float)
            processor = MolmoProcessor.from_pretrained(cfg.model.config_name, trust_remote_code=True)
            model = TransparentMolmo(cfg.model.name, molmo, processor, molmo.device, dtype)
            model = GenerativeWrapper(processor, model, molmo.device, dtype, list(cfg.model.vqa_tokens),
                                      output_attentions=True)

        case "pixtral":
            pixtral = LlavaForConditionalGeneration.from_pretrained(cfg.model.config_name,
                                                                    torch_dtype=dtype,
                                                                    low_cpu_mem_usage=True,
                                                                    device_map='auto',
                                                                    attn_implementation=cfg.model.attn_impl)
            processor = AutoProcessor.from_pretrained(cfg.model.processor_path)
            pixtral.generation_config.pad_token_id = processor.tokenizer.eos_token_id
            model = TransparentPixtral(cfg.model.name, pixtral, processor, pixtral.device, dtype, store_on_cpu=True)
            model = GenerativeWrapper(processor, model, pixtral.device, dtype, list(cfg.model.vqa_tokens),
                                      output_attentions=True)

        case _:
            raise ValueError(f"Unsupported model '{cfg.model.name}'")

    return model


def create_eval_task(cfg):
    sampling_config = GenerationConfig(**cfg.sampling_params)
    match cfg.task.name:
        case "SEED-Bench-2":
            dataset = SEEDBenchSingleImage(cfg.task.task_num, Path(cfg.task.json_path), Path(cfg.task.image_root))
            collate = SEEDBenchCollate()
            wrapper = SEEDBenchSingleImageEval(cfg.prompt.text, cfg.task.eval_method, sampling_config)

        case "WhatsUp":
            if cfg.task.part == 'A':
                json_path = Path(cfg.task.json_path_A)
            elif cfg.task.part == 'B':
                json_path = Path(cfg.task.json_path_B)
            else:
                raise ValueError(f"Unsupported What's Up part '{cfg.task.part}'")
            dataset = WhatsUp(Path(cfg.task.image_root), json_path, permute_options=cfg.task.permute)
            collate = WhatsUpCollate()
            wrapper = WhatsUpEval(cfg.prompt.text, cfg.task.eval_method, sampling_config)

        case "GQA":
            dataset = GQA(Path(cfg.task.test_question_file), Path(cfg.task.img_dir))
            collate = GQACollate()
            wrapper = GQAEval(cfg.prompt.text, cfg.task.eval_method, sampling_config)

        case "UnlabeledCOCO":
            dataset = UnlabeledCoco(Path(cfg.task.img_dir), Path(cfg.task.img_descriptions_file), cfg.task.dataset_from,
                                    cfg.task.dataset_to)
            collate = UnlabeledCocoCollate()
            wrapper = UnlabeledCocoEval(cfg.prompt.text, sampling_config)

        case "MMVP":
            dataset = MMVP(Path(cfg.task.csv_path), Path(cfg.task.img_dir))
            collate = MMVPCollate()
            wrapper = MMVPEval(cfg.prompt.text, cfg.task.eval_method)

        case _:
            raise ValueError(f"Unsupported task '{cfg.task.name}'")

    return dataset, wrapper, collate


def create_callbacks(cfg, eval_wrapper: EvalWrapper, run_folder: Path, model: GenerativeWrapper):
    eval_wrapper.add_callback(ImageBoundariesCallback())
    if cfg.save_full_graphs:
        (run_folder / "graphs").mkdir(parents=True, exist_ok=True)
        # callback = GraphCallback(run_folder / "graphs", cfg.renormalization_threshold, cfg.sparsification_threshold)
        callback = GraphTensorCallback(run_folder / "graphs", head_batch_size=cfg.head_batch_size)
        eval_wrapper.add_callback(callback)
    if cfg.last_token_logit_lens:
        callback = LogitLensCallback(["A", "B", "C", "D"], normalize_lens=cfg.normalize_lens,
                                     tokenizer=model.processor.tokenizer)
        eval_wrapper.add_callback(callback)
    if cfg.capture_residual_streams:
        (run_folder / "residuals").mkdir(parents=True, exist_ok=True)
        callback = ResidualsByLayerCallback(run_folder / "residuals", cfg.num_last_tokens, cfg.from_layer)
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
    if cfg.node_edge_densities:
        graph_metrics.append(NodeEdgeDensities())
    if cfg.cross_modal_edges:
        graph_metrics.append(CrossModalEdges())
    if cfg.global_clustering_coefficient:
        graph_metrics.append(GlobalClusteringCoefficient())
    if cfg.residual_stream_heights:
        graph_metrics.append(ResidualStreamHeights())
    if cfg.centrality_overlap:
        graph_metrics.append(CentralityOverlap())

    return vertex_metrics, graph_metrics
