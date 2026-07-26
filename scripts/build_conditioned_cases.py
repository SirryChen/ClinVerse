#!/usr/bin/env python3
"""生成条件Synthea案例：数据集先验 + 定向Synthea生成。

支持任意已注册数据源（见 core/registry.py），通过 --dataset 选择：

用法：
    python scripts/build_conditioned_cases.py --limit 5
    python scripts/build_conditioned_cases.py --dataset sicdb-ext --limit 5 --copies 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_yaml_config
from core.registry import get_adapter, known_datasets
from pipeline.case_builder import build_cases_conditioned_batch
from pipeline.patient_info_units import OpenAIChatPatientInfoExtractor

REPO_ROOT = Path(__file__).parent.parent
TRAIN_ROOT = REPO_ROOT.parent  # workspace containing the Synthea checkout


def main():
    parser = argparse.ArgumentParser(description="用数据源先验条件生成Synthea患者背景")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", default="mimic-iv-ext",
                        help=f"数据源名称，已注册：{known_datasets()}")
    parser.add_argument("--limit",   type=int, default=5,   help="处理的入院案例数量")
    parser.add_argument("--copies",  type=int, default=10,  help="每个患者生成的Synthea副本数")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="每次 Synthea JVM 处理的 admission 数量")
    parser.add_argument("--resume", action="store_true",
                        help="跳过输出目录中已有 case 文件和索引项的 admission")
    parser.add_argument("--output",  default=None,          help="案例输出目录（覆盖默认）")
    parser.add_argument("--info-extractor-provider", default=None,
                        help="可选 llm_providers key，用于抽取 patient_information_units")
    parser.add_argument("--keep-runtime-artifacts", action="store_true",
                        help="保留每个 admission 的 Synthea fixed records、prior JSON 和 FHIR 输出，用于调试")
    parser.add_argument("--keep-table-cache", action="store_true",
                        help="允许 MIMIC adapter 写入 _tables_cache.pkl；默认禁用以避免额外占用空间")
    args = parser.parse_args()
    if not args.keep_table_cache:
        os.environ["CLINVERSE_DISABLE_TABLE_CACHE"] = "1"
    cfg = load_yaml_config(args.config)

    synthea_root = TRAIN_ROOT / "synthea"
    runtime_dir  = REPO_ROOT / "data" / "runtime"
    module_dir   = REPO_ROOT / "synthea_extensions" / "modules"
    cases_dir    = Path(args.output) if args.output else REPO_ROOT / "data" / "cases"
    info_extractor = None
    if not args.keep_runtime_artifacts:
        for runtime_child in (
            runtime_dir / "fixed_records",
            runtime_dir / "mimic_priors",
            runtime_dir / "conditioned_output",
            runtime_dir / "conditioned_batches",
            runtime_dir / "empty_priors",
        ):
            shutil.rmtree(runtime_child, ignore_errors=True)
    if args.info_extractor_provider:
        llm_cfg = cfg["llm_providers"][args.info_extractor_provider]
        info_extractor = OpenAIChatPatientInfoExtractor(
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg["base_url"],
            model=llm_cfg.get("info_extractor_model")
            or llm_cfg.get("patient_actor_model")
            or llm_cfg["doctor_model"],
            timeout=llm_cfg.get("timeout", 120),
            extra_body=llm_cfg.get("extra_body"),
        )

    print(f"[配置]")
    print(f"  数据源:       {args.dataset}")
    print(f"  Synthea根目录: {synthea_root}")
    print(f"  运行时目录:   {runtime_dir}")
    print(f"  自定义模块:   {module_dir}")
    print(f"  输出目录:     {cases_dir}")
    print(f"  案例数量:     {args.limit}")
    print(f"  每患者副本数: {args.copies}")
    print(f"  JVM批大小:    {args.batch_size}")
    print(f"  断点续跑:     {args.resume}")
    print()

    # 检查 Synthea JAR 是否已编译
    jar = list(synthea_root.rglob("synthea-with-dependencies.jar"))
    if not jar:
        print("❌ 未找到 synthea-with-dependencies.jar")
        print("   请先运行：cd synthea && ./gradlew uberJar")
        sys.exit(1)
    print(f"  Synthea JAR: {jar[0]}")

    # 加载入院记录（含 icd_diagnoses 字段）
    adapter = get_adapter(args.dataset)
    print(f"\n加载 {args.dataset} 入院记录（前 {args.limit} 条）...")
    admissions = list(adapter.load_admissions(limit=args.limit))
    print(f"  已加载 {len(admissions)} 条入院记录")

    sample = admissions[0] if admissions else {}
    n_icd = len(sample.get("icd_diagnoses", []))
    print(f"  样例 icd_diagnoses 数量: {n_icd} 条（含主诊断）")

    print(f"\n开始条件生成（两级：full_conditioned → demographic_conditioned）...\n")

    stats = build_cases_conditioned_batch(
        admissions=admissions,
        adapter=adapter,
        cases_dir=cases_dir,
        synthea_root=synthea_root,
        runtime_dir=runtime_dir,
        extra_module_dir=module_dir,
        top_k=3,
        n_copies=args.copies,
        verbose=True,
        info_extractor=info_extractor,
        cleanup_runtime_artifacts=not args.keep_runtime_artifacts,
        batch_size=args.batch_size,
        resume=args.resume,
    )

    print(f"\n完成！")
    print(f"  处理:                    {stats['processed']}")
    print(f"  full_conditioned:        {stats.get('full_conditioned', 0)}")
    print(f"  demographic_conditioned: {stats.get('demographic_conditioned', 0)}")
    print(f"  跳过（双层均失败）:      {stats['skipped_no_match']}")
    print(f"  已写入案例:              {stats['written']}")
    print(f"  跳过（断点已完成）:      {stats.get('skipped_existing', 0)}")


if __name__ == "__main__":
    main()
