"""Write current dataset documentation and hashes; no automatic ZIP packaging."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/dataset_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    m = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    s = json.loads((OUT / 'statistics.json').read_text(encoding='utf-8'))
    validation = json.loads((OUT / 'validation_report.json').read_text(encoding='utf-8'))
    assert validation['status'] == 'PASS'
    train = [json.loads(x) for x in (OUT / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    selected_sources = json.loads((OUT / 'source_rows_selected.json').read_text(encoding='utf-8'))
    important_files = [
        'configs/dataset_v1.json', 'configs/synthea_candidates_5000.properties',
        'scripts/generate_candidates.ps1', 'scripts/build_dataset.py', 'scripts/validate_dataset.py',
        'scripts/preview_source.py', 'scripts/finalize_dataset.py',
        'tools/downloads.json', 'tools/synthea-with-dependencies.jar',
        'tools/tokenizer/Qwen3-4B/tokenizer.json', 'tools/tokenizer/Qwen3-4B/tokenizer_config.json',
        'data/source/synthea_embedded_defaults.properties', 'logs/synthea_candidates_5000.log',
        'data/preview_100/samples_100.jsonl',
    ]
    reproduction = {
        'date': '2026-10-16', 'host_timezone': 'China Standard Time',
        'java_runtime': 'Eclipse Temurin OpenJDK 21.0.12.1+1-LTS, Windows x64',
        'source_generator_project': 'https://github.com/synthetichealth/synthea',
        'generator_documentation': 'https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running',
        'generator_build': 'd9d07a6', 'generation': m['generation'],
        'source_csv_sha256': m['source_csv_sha256'],
        'workspace_files_sha256': {name: sha(ROOT / name) for name in important_files},
        'steps': ['scripts/generate_candidates.ps1 (requires bundled Java/jar; refuses source overwrite)',
                  'python scripts/build_dataset.py', 'python scripts/validate_dataset.py',
                  'python scripts/finalize_dataset.py'],
        'package_scope': 'Dataset artifacts and selected raw source rows only; full source CSVs, JVM, generator and tokenizer remain in the workspace.',
    }
    dump(OUT / 'reproducibility.json', reproduction)
    lines = ['# 个人医疗档案数据集 v1', '',
             '构建日期：2026-10-16。验收：PASS。全部为Synthea合成患者数据，非真实患者档案。', '',
             '共2,328位不同患者、9,312组字段问答。四个目标字段为姓名、出生日期、家庭住址、诊断。',
             '按用户授权移除FIRST/MIDDLE/LAST及源患者MAIDEN字段末尾数字；保留文字、重音符号及内部空格。完整姓名连接非空分量。生日、地址、诊断不变。',
             '姓名格式为natural_no_numeric_suffix_v1；原带编号数据包已移除，不再提供旧姓名数据集。旧编号姓名检查点和预测不能用于新版结果，需要重新训练。', '',
             '## 文件与划分', '', '| 集合 | 记录数 | 问答数 | 每类诊断记录数 |',
             '| --- | ---: | ---: | ---: |',
             '| 开发 dev | 128 | 512 | 6或7 |', '| 训练 train | 1,000 | 4,000 | 50 |',
             '| 验证 validation | 200 | 800 | 10 |', '| 测试 test | 1,000 | 4,000 | 50 |', '',
             '- [dev.jsonl](dev.jsonl)、[train.jsonl](train.jsonl)、[validation.jsonl](validation.jsonl)、[test.jsonl](test.jsonl)：正式划分。',
             '- [all_records.jsonl](all_records.jsonl)：四个集合按上述顺序合并，split字段保留；不能再与分集合文件拼接计数。',
             '- [训练样例_20条.md](训练样例_20条.md)：训练集中每类诊断一个样例，附完整前缀与四组问答。',
             '- [statistics.json](statistics.json)：字段取值数量、出生年代分布、诊断分布和token长度。',
             '- [validation_report.json](validation_report.json)：独立源字段核对及跨集合交集检查。',
             '- [source_rows_selected.json](source_rows_selected.json)：2,328条所选患者、就诊及诊断的完整原始行，数组顺序与all_records.jsonl一致。',
             '- [manifest.json](manifest.json)、[reproducibility.json](reproducibility.json)、[artifact_hashes.json](artifact_hashes.json)：配置、来源及哈希。', '',
             '## 目标字段与输入', '', '| 目标 | 原字段 | 规则 |', '| --- | --- | --- |',
             '| name | patients.FIRST/MIDDLE/LAST | 非空分量以空格拼接，保留原分量 |',
             '| birth_date | patients.BIRTHDATE | 原值 |',
             '| address | patients.ADDRESS | 原街道住址；不补充城市、州、邮编 |',
             '| diagnosis | conditions.DESCRIPTION | 原英文描述及括号 |', '',
             '仅private_prefix用于私有记录编码。患者Id、就诊Id、就诊时间和诊断CODE留在provenance/source_rows中，不进入问题或公共文本。',
             'qa.answer仅用于训练监督或离线评分；测试自由生成不提供正确答案。每道问题独立读取同一份公共文本缓存。', '',
             '## 来源与筛选', '',
             '使用官方Synthea生成器build d9d07a6、固定种子2026101601、医生种子2026101602、参考日和结束日2026-10-16、Massachusetts默认人群，生成5,000名患者。',
             '原始CSV包含5,000名患者、714,070次就诊和427,320行condition。完整病史导出，仅输出patients/encounters/conditions三个CSV。',
             '70位患者没有符合条件的关联疾病诊断；其余4,930位形成可用候选代表。按姓名或住址的规范化碰撞建立连通分组，一组只选一位，不改原值。',
             '所有正式集合都排除了先前100条预览中出现的患者Id、完整姓名或住址。候选中未发生此类重叠。',
             '选取按独立候选患者覆盖量排序且数量足够的20个疾病诊断描述，以二分图最大匹配满足诊断配额。标签表依据来源数量确定，未使用模型成绩。',
             '每位患者仅选一个诊断及其关联就诊，不称为主诊断。正式20类为样本充足类别，不承诺保留预览中的全部稀有诊断。',
             f'所选患者中有{sum(bool(r["DEATHDATE"]) for r in selected_sources["patients"])}位源记录注明已死亡；本数据是个人与历史医疗信息，不代表全部为当前在诊或存活患者。', '',
             '## 隔离与分布', '',
             '四个集合之间：患者Id、完整姓名、住址、分组Id交集均为0。生日和诊断允许重复。',
             '每类诊断内按生日排序并作加权交错划分，随后对各集合独立打乱，以兼顾精确数量和年龄分布。',
             '原始生成器关联保留，不能宣称所有字段独立。类别均衡是受控实验设计，不代表自然患病率。', '',
             '## Qwen3-4B token长度', '',
             '采用工作区已有Qwen3-4B tokenizer快照，使用tokenizers 0.22.1；文件哈希见manifest。',
             '以下只统计原始private_prefix，不含聊天模板、共享引导U或公共文本C；所有答案也分别记录了实际token长度。', '',
             '| 集合 | 最短 | 均值 | P95 | 最长 |', '| --- | ---: | ---: | ---: | ---: |']
    for split, data in s.items():
        t = data['private_prefix_tokens']
        lines.append(f"| {split} | {t['min']} | {t['mean']} | {t['p95']} | {t['max']} |")
    lines += ['', '## 20类原始诊断', '']
    lines += [f'- {label}' for label in m['diagnosis_selection']['labels']]
    lines += ['', '## 当前边界', '',
              '本次完成数据构建、精确源值核对、划分与token长度统计。尚未训练引导或运行缓存恢复实验，未生成修改源值的反事实样本。',
              '后续用验证集选择配置，测试集不参与适应性调参。恢复率是否改善、是否通过缓存交换和中性前缀对照，仍需实验验证。', '',
              '## 复现', '',
              '在工作目录中，现有源CSV可直接运行python scripts/build_dataset.py，随后运行python scripts/validate_dataset.py和python scripts/finalize_dataset.py。',
              '候选源CSV已冻结；生成脚本拒绝覆盖它们。需要重新生成时，应明确使用新输出路径，并保留原源快照。',
              '生成器、Java和tokenizer位于工作目录tools/，完整源CSV位于data/source/candidates_5000/csv/，日志位于logs/。',
              '直接使用当前数据集目录，不自动生成ZIP或带编号备份。', '',
              '[Synthea官方项目](https://github.com/synthetichealth/synthea) · [生成说明](https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running)', '']
    (OUT / '数据集说明.md').write_text('\n'.join(lines), encoding='utf-8')
    examples = ['# 训练集样例：每类诊断一条', '',
                '以下20条均属于train，原值不翻译、不替换；不会增加训练样本量。', '']
    for label in m['diagnosis_selection']['labels']:
        r = next(r for r in train if r['fields']['diagnosis'] == label)
        examples += [f'## {r["record_id"]}', '', '```text', r['private_prefix'], '```', '',
                     '| Field | Question | Answer |', '| --- | --- | --- |']
        examples += ['| ' + ' | '.join([q['field'], q['question'], q['answer']]) + ' |' for q in r['qa']]
        examples.append('')
    (OUT / '训练样例_20条.md').write_text('\n'.join(examples), encoding='utf-8')
    dump(OUT / 'artifact_hashes.json', {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps({'dataset': str(OUT), 'packaging': 'disabled', 'name_format': m['name_format']}))


if __name__ == '__main__':
    main()
