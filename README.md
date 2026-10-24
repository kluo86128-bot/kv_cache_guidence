# 医疗隐私引导实验

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
