# 医疗隐私引导实验

HR 分段 U 训练分支（2026-10-27）：新增 `configs/experiment_v14_hr_split_u.json`、`configs/evaluation_v10_hr_split_u.json`，场景 hr，`loss.type=field_focal_split`；Focal 字段固定为 identity_number、email、address、birth_date、hire_date、leave_start，gamma=2、alpha=1；其余8字段CE。新数据 `data/dataset_multiscene_v3_hr_balanced`、独立输入缓存 `data/input_cache_hr_v3_balanced`。共享U48=CE U32+Focal U16全部从头初始化，一次P+U+C56编码、只保留C KV。姓名和生日组件先平均为原字段；两个目标各按记录中实际有效原字段数归一化，各自对整份U求导后截取CE前32/Focal后16；两段独立Adam状态，整批缺监督时对应grad=None，统一裁剪。入职/请假开始日期保持完整字符串。评估NLL仍为普通CE。

在Linux服务器同步新版代码、两份新HR配置和数据后，先执行 `python scripts/prepare_scene_input_caches.py --dataset-directory data/dataset_multiscene_v3_hr_balanced --scene hr --output-directory data/input_cache_hr_v3_balanced`，再执行 `bash scripts/train_hr_split_u_background.sh 0,1,3`（GPU列表按实际资源分配）。脚本自动定位项目，使用当前环境Python或MEDICAL_PRIVACY_PYTHON，启动前CPU检查数据/缓存哈希与本地模型目录，不加载模型。默认模型 `/home/kevin/models/Qwen3-4B`；若在A800服务器使用 `/data1/kevin/model/Qwen3-4B`，只改两份新HR配置里的模型路径。默认100轮、batch8、seed17、LR0.01、最后一轮验证并自动测试，可加--epoch-validation。

训练目录 `runs/soft_hr_v3_balanced_split_u32_16_focal6_seed17_100epochs`，日志 `logs/soft_hr_v3_balanced_split_u32_16_focal6_seed17_100epochs.log`；最终测试目录 `runs/test_hr_v3_balanced_split_u32_16_focal6_best`。暂停用 `python training_control.py pause --run-root runs/soft_hr_v3_balanced_split_u32_16_focal6_seed17_100epochs --wait`，恢复用 `python training_control.py recover --run-root runs/soft_hr_v3_balanced_split_u32_16_focal6_seed17_100epochs --gpus 0,1,3`。阶段测试用evaluation_v10_hr_split_u.json、同run的latest.pt、--interim-test及独立输出目录。不能把旧HR数据或其他方案断点作为新实验起点。医疗和金融配置、损失分支和输出目录保留；同步medical_kv/与resume_compatibility_v1.json以启用经回归验证的代码迁移。181项相关CPU测试通过；四条旧医药/金融分段U及PCGrad的修改前CPU断点恢复后，U、Adam和调度器与旧代码连续训练精确一致，29份旧训练/评估配置原样保留。此分支只做CPU/tiny验证，正式CUDA/NCCL未在本机运行。

HR 新版数据（2026-10-27）：`configs/dataset_hr_v3_balanced.json`、`scripts/build_hr_balanced_dataset.py` 构建独立目录 `data/dataset_multiscene_v3_hr_balanced`，保持原记录 ID、划分、顺序和医药/金融记录；HR 保留姓名、出生日期、薪资、入职日期、部门、职位、身份号码、住址、邮箱，以及请假开始、时长、原因、投诉对象、投诉内容共14字段。五个低频字段在训练/验证/测试分别覆盖1000/200/1000条记录，结构化和自然叙述各占一半。现有字段值和清理后的原始正文保留，只为缺项追加业务事实；旧补充段重写为自然业务文本。来源标记只在元数据，private_prefix和提问不包含supplemental synthetic、来源名或审计ID。

来源快照按 revision 和 SHA256 固定在 `data/source/hr_augmentation_v1`。实际供体为 HR-MultiWOZ（Apache-2.0，引用原论文并说明公开重实现）和 Consumer Dividends WorkplaceDispute（CC BY-NC 4.0，非商业研究且保留署名）；Realigns 仅缓存审计，未作为供体，EEOC未使用。源事件先按去人名/数字的内容与三词组Jaccard相似度分组，再整体划分；同集合允许均衡复用，常见答案跨集合重复单独报告，不等同于来源事件泄露。投诉对象不直接照搬“涉及人员”列表：去掉报告人，优先取描述中的行为主体；职场对话要求有具体行为陈述和明确对话对象。生成的跨来源员工关联明确标为合成，不声称原来源的真实人物关联或已验证指控。

执行 `python scripts/build_hr_balanced_dataset.py`，已有版本默认拒绝覆盖；`--rebuild`核对已发布版本哈希，先完整校验新构建，再保留旧派生版本并发布。执行 `python scripts/validate_hr_balanced_dataset.py`复核原记录/原值、数量、字面证据和来源隔离。报告位于新目录的 `hr_augmentation_report.json`、`hr_source_inventory.json`、`validation_report.json`、`statistics.json`。原版中日期不支持成年员工事件的记录不新增事件，原值不改；自动检查及少量抽查不代表全量人工语义审查。所有旧实验仍使用原v2目录。

新 HR 输入 token 缓存请在 Linux 服务器执行：`python scripts/prepare_scene_input_caches.py --dataset-directory data/dataset_multiscene_v3_hr_balanced --scene hr --output-directory data/input_cache_hr_v3_balanced`。它只分词，不加载模型或生成KV；新数据拒绝覆盖旧v2默认缓存。数据构建和缓存准备入口供上文HR分段配置使用，不能拿旧输入断点直接续训。`data/`被gitignore排除，新数据需要单独同步，或同步冻结v2父数据后在服务器复现构建。

新增金融 PCGrad：`configs/experiment_v13_finance_pcgrad.json`、`configs/evaluation_v9_finance_pcgrad.json` 使用独立 `loss.type=field_focal_pcgrad`。共享 U32 全部从 seed17 的 token embedding 初始化，五个 Focal 字段仍为 timestamp、identity_number、card_number、account_balance、transfer_amount，其余字段 CE；gamma=2、alpha=1。每条记录先按有效答案 token（含 EOS）求均值，再在原始字段内平均组件，最后按有效原始字段等权；缺字段不补齐。每个原始字段是一个 PCGrad 任务，逐字段跨卡 SUM 后再投影，修正梯度相加、全局裁剪、AdamW 更新完整 U32。每记录一次 P+U32+C56 编码，回答只读取一份 C KV。评估 NLL 仍为普通 CE，数据和旧医药/金融实验配置保持。金融分段 U48 与此实验同时改变容量和更新方式，二者比较不能单独归因为 PCGrad。

在新 A800 服务器 `/data1/kevin/medical_privacy` 激活 `medical_privacy` 环境，先同步更新的 `medical_kv/`、`scripts/`、本实验两份配置和 `configs/resume_compatibility_v1.json`；保留服务器已有医药配置中的本地模型路径。新配置模型路径已设为 `/data1/kevin/model/Qwen3-4B`。执行 `bash scripts/train_finance_pcgrad_background.sh 1,2`，使用物理 GPU 1、2，全局 batch8、LR0.01、100轮，默认最后一轮验证并自动测试。若这两张卡仍承载医药训练，先用 `python training_control.py pause --run-root runs/soft_medical_v2_clean_pcgrad_u32_seed17_100epochs --wait` 安全暂停以释放资源；同服务器并行运行需分别安排资源。不要使用旧服务器写死路径的启动脚本。脚本自动定位项目、使用当前 Python，已有新实验 run/log 时拒绝覆盖。

新训练与日志分别为 `runs/soft_finance_v2_clean_pcgrad_u32_focal5_seed17_100epochs` 和 `logs/soft_finance_v2_clean_pcgrad_u32_focal5_seed17_100epochs.log`；自动测试输出 `runs/test_finance_v2_clean_pcgrad_u32_focal5_best`。日志 `loss=ce_other_loss+focal_loss` 为投影前数值，`focal_group_loss` 为每记录可用 Focal 原字段均值、无则0；`pcgrad_active_fields`/`pcgrad_projections`/`pcgrad_unprojected_norm`/`pcgrad_projected_norm` 沿用医药 PCGrad 定义。这里没有独立 U16 或分段梯度范数。暂停/续训使用 `python training_control.py pause --run-root runs/soft_finance_v2_clean_pcgrad_u32_focal5_seed17_100epochs --wait` 和 `python training_control.py recover --run-root runs/soft_finance_v2_clean_pcgrad_u32_focal5_seed17_100epochs --gpus 1,2`，恢复入口自动识别金融 PCGrad。禁止从其他方案 checkpoint 跨目标续训。仅完成 CPU/tiny 验证，未运行正式 Qwen3-4B/CUDA/NCCL 实验。

新增金融分段 U：`configs/experiment_v12_finance_split_u.json`、`configs/evaluation_v8_finance_split_u.json` 使用独立 `loss.type=field_focal_split`。Focal 字段固定在配置列表 `timestamp`、`identity_number`、`card_number`、`account_balance`、`transfer_amount`，gamma=2、alpha=1；其余字段CE。五个字段各占一份原始字段权重：每记录 `(其他字段CE之和+实际存在的选中字段Focal之和)/有效原始字段数`，不把五字段平均成一个字段，不补缺失目标，姓名组件先平均，包含EOS。前CE U32与后Focal U16全部从头初始化，同一份U48+C56编码得到公共C KV，分别求导并选取两段梯度后统一裁剪，独立Adam状态；整批缺监督时对应段跳过动量更新。金融训练不会读取或更新医疗U。医疗旧配置、`birth_focal_split`损失、`birth_length`键和日志口径保持；正式CUDA/NCCL仍需在服务器验证。

在项目根目录激活 `medical_privacy` 环境后执行 `bash scripts/train_finance_split_u_background.sh 0,1,3`（替换成分配给本实验的GPU）。新脚本按自身位置定位项目，并使用当前环境Python，可通过 `MEDICAL_PRIVACY_PYTHON` 指定解释器，不依赖原服务器的固定项目/环境路径。两份金融配置默认模型仍为 `/home/kevin/models/Qwen3-4B`；若新服务器模型在 `/data1/kevin/model/Qwen3-4B`，只修改上述两份新配置的 `model_name_or_path`，无需改医疗配置。金融目录 `runs/soft_finance_v2_clean_split_u32_16_focal5_seed17_100epochs`，日志同名位于 `logs/`，100轮最后验证后自动测试至 `runs/test_finance_v2_clean_split_u32_16_focal5_best`；可加 `--epoch-validation`。沿用冻结v2 clean数据、输入token缓存、seed17、全局batch8、LR0.01，不启用PCGrad或离散搜索。

金融日志 `loss=ce_other_loss+focal_loss`，两项均含有效原字段归一化；`ce_loss`为全字段普通CE对照；`focal_group_loss`为每记录可用Focal原字段均值（无则0）再平均记录；`focal_fields_per_record`为每记录可用Focal字段数的batch均值。`other_gradient_norm`、`focal_gradient_norm`为统一裁剪后的两段梯度范数，`gradient_norm`为裁剪前总范数；评估NLL始终普通CE。暂停/恢复分别使用 `python training_control.py pause --run-root runs/soft_finance_v2_clean_split_u32_16_focal5_seed17_100epochs --wait`、`python training_control.py recover --run-root runs/soft_finance_v2_clean_split_u32_16_focal5_seed17_100epochs --gpus 0,1,3`。旧医疗恢复需同步新版 `medical_kv/` 与 `configs/resume_compatibility_v1.json`，旧医疗配置无需更改；兼容迁移不允许更改损失、字段列表、数据或模型身份后继续原断点。

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
