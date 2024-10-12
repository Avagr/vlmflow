import os
from pathlib import Path

from graph_tool import Graph  # noqa
import hydra
import wandb
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from utils.eval import eval_model, merge_results
from utils.misc import set_random_seed, count_parameters, timestamp
from utils.setup import create_model, create_eval_task, create_callbacks
from torch.backends import opt_einsum


@hydra.main(config_path="configs", config_name="eval_config", version_base=None)
def run(cfg: DictConfig):

    torch.autograd.set_detect_anomaly(cfg.detect_anomalies, check_nan=True)
    torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf32
    torch.backends.cudnn.allow_tf32 = cfg.use_tf32
    torch._dynamo.config.cache_size_limit = 16

    set_random_seed(cfg.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(OmegaConf.to_yaml(cfg))
    if cfg.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"


    if cfg.resume_wandb_id is None:
        run_id = wandb.util.generate_id()
    else:
        run_id = cfg.resume_wandb_id

    print(f"Run id: {run_id}")

    choices = HydraConfig.get().runtime.choices
    root_dir = Path(__file__).parent.resolve()
    prompt_path = root_dir.joinpath(root_dir, "configs", "prompts",
                                    f"{choices['model']}", f"{choices['task']}_{cfg.task.eval_method}.yaml")
    cfg.prompt = OmegaConf.load(prompt_path)

    print("Prompt:\n" + cfg.prompt.text)

    run_folder = Path(cfg.results_path) / f"{cfg.task.display_name}" / f"{cfg.name}_{timestamp()}"
    run_folder.mkdir(parents=True, exist_ok=True)

    dataset, wrapper, collate_fn = create_eval_task(cfg)
    model = create_model(cfg)

    cfg.model_size = count_parameters(model)
    wandb.init(id=run_id, resume="must" if cfg.resume_wandb_id is not None else "never",
               project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.name, group=cfg.task.display_name,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb.watch(model)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=cfg.pin_memory, collate_fn=collate_fn)

    callbacks = create_callbacks(cfg, wrapper, run_folder, model)



    with torch.inference_mode(mode=not cfg.need_grad):
        accuracy, results = eval_model(model, wrapper, loader, cfg.show_tqdm)

    merged_results = merge_results(results, callbacks)

    save_file = run_folder / f"results_table.parquet"
    df = merged_results.to_table(dataset).get_dataframe()
    df.attrs["run_id"] = run_id
    df.to_parquet(save_file)
    if cfg.disable_wandb:
        print({"accuracy": accuracy, })
    else:
        wandb.log({"results_table": merged_results.to_table(dataset)})
        wandb.run.summary.update({"accuracy": accuracy})


if __name__ == '__main__':
    run()
