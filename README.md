# 医疗隐私引导实验

新增 PCGrad 独立对照：`configs/experiment_v11_medical_pcgrad.json` 使用共享 U32，从固定种子选择的 token embedding 完全从头训练；生日仅 Focal（gamma=2、alpha=1），其他字段 CE。姓名、出生日期内部组件先聚合，按每记录有效原始字段等权。按原始字段分别求对完整 U32 的梯度，跨卡逐字段汇总后，按 [PCGrad Algorithm 1](https://arxiv.org/html/2001.06782v4#S2.SS3) 对负点积梯度投影，投影参照始终是未修改的原始字段梯度。修正梯度相加后统一裁剪并用 AdamW 更新；没有额外字段均值或梯度单位化。P+U32+C56 每条记录只生成一份公共 C KV，全部字段共用。保留分段 U48 实验；比较投影效果应使用同 U32、相同损失的 `experiment_v9_medical_birth_focal.json` 对照。多任务投影与 Adam 更新不保证每个字段单调改善。

在 Linux 服务器同步 `medical_kv/`、`configs/`、`scripts/` 和 `training_control.py`，执行 `bash scripts/train_medical_pcgrad_background.sh 0,1,3`。U32/LR0.01/全局batch8/seed17/100轮及冻结数据保持，默认第100轮验证后自动测试，可加 `--epoch-validation`。新训练目录 `runs/soft_medical_v2_clean_pcgrad_u32_seed17_100epochs`，日志 `logs/soft_medical_v2_clean_pcgrad_u32_seed17_100epochs.log`，测试 `runs/test_medical_v2_clean_pcgrad_u32_best`。日志 loss 是投影前 `ce_other_loss+focal_loss`，另报 `pcgrad_active_fields`、`pcgrad_projections`、`pcgrad_unprojected_norm`、`pcgrad_projected_norm`；评估 NLL 仍为 CE。暂停和恢复使用 `python training_control.py pause --run-root runs/soft_medical_v2_clean_pcgrad_u32_seed17_100epochs --wait`、`python training_control.py recover --run-root runs/soft_medical_v2_clean_pcgrad_u32_seed17_100epochs --gpus 0,1,3`。投影随机顺序由种子、已完成步数及字段确定，恢复保留本实验 U/Adam/调度器和进度；禁止旧损失断点跨目标续训。训练需要逐字段反向，额外开销需在服务器测量；目前仅 CPU/tiny 实现验证，未启动正式 CUDA 作业。

最新分段 U 实验使用 `configs/experiment_v10_medical_split_u.json`：完整 U48 由其他字段 U32 和日期 U16 拼接，两段全部从固定种子选择的模型 token embedding 初始化，从头训练，不加载任何旧实验 U。完整 U48 与 P、C56 一次编码产生一份公共 C KV。其他字段 CE 只更新前32位置，生日 Focal（gamma=2、alpha=1）只更新后16位置；两段使用独立 Adam 状态，缺监督时跳过对应段的动量更新。原始字段等权聚合保持，日志 `loss=ce_other_loss+focal_loss` 仅是两项目标的数值合计，实际梯度分别路由。验证/测试 NLL 仍为普通 CE。公共缓存仍可能带来字段间接影响。

在 Linux 服务器同步 `medical_kv/`、`configs/`、`scripts/` 和 `training_control.py`，执行 `bash scripts/train_medical_split_u_background.sh 0,1,3`。不需要旧检查点参数，已移除 `--initialize-from`。配置显式标记 `prompt_initialization=fresh_token_embeddings`，两段 U、优化器、调度器和训练步数均从头开始。完整初始 U 和初始化方式保存到断点/元数据。新目录 `runs/soft_medical_v2_clean_split_u32_16_fresh_seed17_100epochs`，默认100轮最后验证并自动测试到 `runs/test_medical_v2_clean_split_u32_16_fresh_best`，可加 `--epoch-validation`。日志 `logs/soft_medical_v2_clean_split_u32_16_fresh_seed17_100epochs.log`。暂停/恢复分别使用 `python training_control.py pause --run-root runs/soft_medical_v2_clean_split_u32_16_fresh_seed17_100epochs --wait` 和 `python training_control.py recover --run-root runs/soft_medical_v2_clean_split_u32_16_fresh_seed17_100epochs --gpus 0,1,3`；恢复仅续本实验进度，配置校验拒绝旧初始化方案。正式 Qwen3-4B/CUDA 作业仅在服务器运行。

最新医疗字段分配：`configs/experiment_v9_medical_birth_focal.json` 使用生日仅 Focal（gamma=2、alpha=1）、其他字段仅 CE，按有效原始字段等权平均，保留 EOS 和生日组件均值。生日不再叠加 CE，也没有 0.5 附加权重。日志 `loss = ce_other_loss + focal_loss`，两项含原字段归一化；`ce_loss` 仅作为全字段 CE 对照记录，评估 NLL 仍为 CE。同步 `medical_kv/`、`configs/`、`scripts/` 后在服务器执行 `bash scripts/train_medical_birth_focal_background.sh 0,1,3`，新目录 `runs/soft_medical_v2_clean_birth_focal_seed17_100epochs` 从头训练，默认第 100 轮验证并自动测试至 `runs/test_medical_v2_clean_birth_focal_best`。通用暂停/恢复入口使用新 run-root，旧实验继续使用原配置和目录。

当前医疗联合损失：`configs/experiment_v8_medical_ce_focal_birth.json` 使用 `(各有效字段 CE 之和 + 0.5 * 出生日期 Focal) / 有效字段数`，生日 Focal 为年/月/日三问的 token 均值再平均，gamma=2、alpha=1，包含 EOS。缺少生日的记录仅计算原 CE。`0.5` 是未调优起始值，联合损失不保证各字段精确匹配率单调改善。沿用数据、U32、公共 C56、全局 batch8、LR0.01 和 100 轮；新目录从头训练，旧 CE/Focal 结果保留。服务器后台入口 `bash scripts/train_medical_joint_background.sh 0,1,3`，默认第 100 轮验证后自动测试，可加 `--epoch-validation`。日志保存总 loss、ce_loss、按原字段权重缩放的 focal_loss、birth_date_focal_loss、focal_weighted_loss；测试 nll 仍是 CE。暂停恢复指定 `runs/soft_medical_v2_clean_ce_focal_birth_w05_seed17_100epochs`，系统自动选择联合配置。

当前医疗 Focal 对照：`configs/experiment_v7_medical_focal.json` 使用纯 token Focal（gamma=2、alpha=1），包含 EOS，保留组件/原始字段等权聚合。训练日志同时记录 `focal_loss` 与 `ce_loss`；验证和测试的 `nll` 仍为普通 CE。新实验从头训练，输出 `runs/soft_medical_v2_clean_focal_gamma2_seed17_100epochs`，沿用 v2 clean 数据及输入 token 缓存。

服务器同步 `medical_kv/`、`configs/`、`scripts/` 和 `training_control.py` 后，在 `/data/kevin/medical_privacy_prompt` 执行 `bash scripts/train_medical_focal_background.sh 0,1,3`。默认跳过中间逐轮验证，第 100 轮验证后自动测试，结果在 `runs/test_medical_v2_clean_focal_gamma2_best`。末尾加 `--epoch-validation` 可逐轮验证。暂停/恢复仍使用 `training_control.py pause/recover --run-root runs/soft_medical_v2_clean_focal_gamma2_seed17_100epochs`（恢复指定 `--gpus`）；不得用旧 CE 断点续训为 Focal。实现只在 CPU 小模型上验证，正式 Qwen3-4B/CUDA 实验在服务器运行。

当前所有活动数据均使用无数字后缀的合成姓名，姓名格式为`natural_no_numeric_suffix_v1`。按用户要求直接修改既有数据，不保留带编号姓名的训练/测试数据备份，旧数据ZIP已移除。

## 数据

- [基础数据集说明](data/dataset_v1/数据集说明.md)：开发128、训练1000、验证200、测试1000；患者及划分保持不变。
- [训练集](data/dataset_v1/train.jsonl)、[验证集](data/dataset_v1/validation.jsonl)、[测试集](data/dataset_v1/test.jsonl)。
- [诊断重叠10%的测试版本](data/dataset_diagnosis_overlap10/数据集说明.md)：同1000位测试患者，100个训练已见诊断、900个未见诊断；姓名同步自然化，其他字段与先前该版本一致。
- [100条预览](data/preview_100/样本预览_100条.md)、[10条预览](data/preview/样本预览_10条.md)：均已去除姓名后缀数字。
- [姓名迁移核验](data/name_naturalization_report.json)：处理范围、旧/新源文件哈希、集合隔离、未改变的字段及已移除的数据包。

只删除FIRST/MIDDLE/LAST及源患者MAIDEN末尾的ASCII数字，保留文字、重音符号和内部空格。生日、地址（含门牌号）、诊断（含分型数字）、患者ID不改。姓名答案、private_prefix、姓名分量、选中源患者行、token长度及哈希同步更新。去号后各集合之间及与预览之间无重名。

正式源CSV位于data/source/candidates_5000/csv；patients.csv也已按此规则更新，encounters.csv和conditions.csv字节未变。官方预览源已转为data/source/preview_csv的三张CSV，带编号原ZIP不再保留。原始下载/生成快照仅保留哈希及变换说明，不保留编号姓名数据副本。后续生成配置generate.append_numbers_to_person_names=false。

## 当前代码与运行

诊断对比训练使用[experiment_v3_diagnosis_contrastive.json](configs/experiment_v3_diagnosis_contrastive.json)，流程和后台命令见[诊断对比训练说明](诊断对比训练说明_v1.md)。五类各建最多五个候选，每患者每轮抽取各类最多一个，支持可复现的跨患者/跨轮次变化；拼写支持1至3次多位置字符操作。语义替换只能使用实际训练患者历史中完整存在且通过测试排除的诊断，规则在[diagnosis_negative_rules.json](configs/diagnosis_negative_rules.json)。**当前来源约束下，第②③④类均缺项；实际每轮细节遗漏550条、拼写1000条，共1550条，不能视为五类全覆盖。** 新输出目录runs/soft_natural_names_diag_pools_seed17_100epochs，需同步源conditions.csv。原数据和v2配置不改，weight/margin均为0.2，仍自由生成六问。

[使用说明及后台命令](实验代码使用说明_v3.md)；[训练配置](configs/experiment_v2_birth_parts.json)；[诊断10%评估配置](configs/evaluation_diagnosis_overlap10.json)。配置统一位于configs/。

默认100轮、关闭早停、全局batch 8；生日拆为年/月/日，姓名仍以完整姓名训练，未自动实施姓名拆问。六问合批，只更新共享U32，回答时仅保留C56缓存；四隐私组等权。离散文本搜索保持关闭。

姓名输入格式已改变，旧编号姓名检查点不能恢复当前训练。允许使用旧best.pt加`--allow-name-format-transfer`评估新测试集，结果记录`name_format_transfer=true`、训练及评估姓名格式；其分数衡量旧U对自然姓名输入的迁移表现。模型、分词器、缓存协议及生日提问模式仍需一致，正式测试仍要求完整训练的验证选优检查点。自然姓名格式匹配的训练应从头开始，新默认目录为runs/soft_natural_names_birth_parts_seed17_100epochs。历史预测、日志及检查点未改写，标记见runs/LEGACY_NAME_FORMAT.json。

使用Linux/CUDA服务器/data/kevin/medical_privacy_prompt，独立环境medical_privacy，模型/home/kevin/models/Qwen3-4B。支持--gpus 0,1等逗号分隔编号；后台命令采用nohup setsid。评估卡数可与训练不同，每卡各加载完整模型。

保存best.pt、final.pt、best_first20.pt、last_first20.pt及每步latest.pt。每轮保存全部验证回答CSV、JSONL、汇总和元数据，checkpoint_results.json记录对应文件。保持四字段主指标，年月日拼接后对完整日期评分。每200位验证患者1400行，每1000位测试患者7000行。

## 复核与重建

- scripts/validate_dataset.py：只读复核基础版字段、源关联、token长度和集合隔离；显式--write-report才更新报告及其哈希。
- scripts/build_diagnosis_overlap.py --validate-only：只读复核当前诊断10%版本。
- scripts/prepare_preview.py及scripts/prepare_preview_100.py：从去号后的预览CSV重建预览。
- scripts/naturalize_existing_names.py：本次一次性迁移脚本，成功后拒绝再次运行，防止覆盖既有记录。
- scripts/finalize_dataset.py：生成数据说明和样例，不自动创建ZIP。

评估应注明使用旧编号姓名训练结果还是自然姓名重新训练结果；随机CPU tiny模型只用于实现验证。数据分组规范见[数据集规范](数据集规范_防止答案记忆_v1.md)。
