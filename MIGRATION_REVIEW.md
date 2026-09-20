# CL_CoTNav 迁移说明与待确认项

本次迁移以 CL_CoTNav 的提交 a55ff1b 为基线，只迁移 Goal2Pixel 的训练数据
rollout、数据整理和 VLM 微调流程。源仓库工作区中的未提交修改没有复制，以免
把 rebuttal 阶段的新实验混入公开代码。

## 已迁移

- scripts/scripts_for_generating_data/：trajectory、pixel、history rollout，
  annotation 调整、合并和 InternVL JSONL 重建。
- scripts/agent/ 中保留 Goal2Pixel 使用的 base_agent.py、
  if_map_generate_data.py，以及允许公开的 evaluation 辅助文件 if_map_eval.py。
- scripts/policy/、scripts/run_utils/：history 表示、建图和图像处理代码。
- InternVL_cleaned/ 中除 eval/ 以外的数据集、模型、patch 和训练代码。
- configs_hydra/、configs_vlm/、task_patch/、il_patch/ 和训练/rollout
  shell 入口。
- 定制 habitat-lab 子模块，固定在提交
  4c536fd11222556c030de14ead97e29420a60823。
- find_scene_name.py：虽然它在清理提交中被误删，但 pixel rollout 仍直接
  导入其中的场景列表，因此从清理前版本恢复。

## 明确未迁移

- InternVL_cleaned/internvl_chat/internvl_cleaned/eval/：按要求排除。源基线
  中有 mp3d_no_traj.py、mp3d_traj.py、quickly_ask.py、utils.py；源工作区
  另有 rebuttal 相关的 md3d_nonpriv_local_controller.py 和
  md3d_offline_pixel.py。
- run_vlm_evaluate.sh：只负责调用上述 evaluation 代码，单独公开无法运行。
- 源仓库所有未提交改动和新增文件，包括 compare_rebuttle_gtrotate.py、
  astar_implemnetation.md、implementation.md、run2.sh、
  select_valid_episodes_by_sr.py，以及被修改的 README、evaluation runner 和
  InternVL eval 注册代码。
- 非 Goal2Pixel agent：if_map.py、if_map_pano.py、if_map_RadioNav.py、
  if_map_SysNav.py、if_map_memory_nav.py、replay_dual_mem.py。
- all_log/、training_data/、evaluate_vis/、预训练权重、cache、build、
  IDE 配置等运行产物或本地环境文件。

## 已确认保留

- scripts/agent/if_map_eval.py：evaluation 辅助文件允许公开。
- configs_hydra/ 中带 valunseen 或 eval 的配置。
- task_patch/ 中的 MP3D、HM3D、OVON 和统计/可视化辅助脚本。

## 暂留在源仓库、需要作者判断

- README_.md：旧版/备份文档，与当前公开 README 内容重复且包含已排除的
  evaluation 说明。
- run1.sh：个人实验快捷命令，当前基线内容不构成通用入口。

## 发布前建议

1. 确认 README_.md 和 run1.sh 是否需要在后续版本公开。
2. 在具备 Habitat 数据和 GPU 环境的机器上完成一次最小 rollout 与单步训练。
3. 添加适合公开发布的 LICENSE，并确认定制 Habitat 代码的第三方许可说明。
