from pathlib import Path
import json


OUTPUTS_DIR = Path("outputs")

with open(OUTPUTS_DIR / "retrieval_comparison.json", "r", encoding="utf-8") as file:
    data = json.load(file)

bm25 = data["bm25"]
dense = data["dense"]

output_path = OUTPUTS_DIR / "manual_evaluation.md"

lines = [
    "# W7D2 Manual Retrieval Evaluation",
    "",
    "## Evaluation Method",
    "",
    "The same 10 questions were evaluated against five indexed PDF documents.",
    "Each question was assigned an expected relevant source document.",
    "Top-1 precision measures whether the expected document was ranked first.",
    "Top-3 hit rate measures whether the expected document appeared within the top three results.",
    "",
    "## Results",
    "",
    "| Q | Expected Source | BM25 Rank | BM25 Top-1 | Dense Rank | Dense Top-1 |",
    "|---|---|---:|---|---:|---|",
]

for b, d in zip(bm25["questions"], dense["questions"]):
    lines.append(
        f"| {b['question_number']} | "
        f"{b['expected_source']} | "
        f"{b['expected_rank']} | "
        f"{'PASS' if b['top1_hit'] else 'MISS'} | "
        f"{d['expected_rank']} | "
        f"{'PASS' if d['top1_hit'] else 'MISS'} |"
    )

lines.extend(
    [
        "",
        "## Summary",
        "",
        f"- BM25 Top-1 Precision: {bm25['metrics']['top1_precision'] * 100:.1f}% "
        f"({bm25['metrics']['top1_hits']}/{bm25['metrics']['total_questions']})",
        f"- BM25 Top-3 Hit Rate: {bm25['metrics']['top3_hit_rate'] * 100:.1f}% "
        f"({bm25['metrics']['top3_hits']}/{bm25['metrics']['total_questions']})",
        f"- Dense Top-1 Precision: {dense['metrics']['top1_precision'] * 100:.1f}% "
        f"({dense['metrics']['top1_hits']}/{dense['metrics']['total_questions']})",
        f"- Dense Top-3 Hit Rate: {dense['metrics']['top3_hit_rate'] * 100:.1f}% "
        f"({dense['metrics']['top3_hits']}/{dense['metrics']['total_questions']})",
        "",
        "## Manual Observation",
        "",
        "BM25 retrieved the expected document within the top three results for all 10 questions.",
        "For Question 2, the expected machine_learning.pdf was ranked second by BM25.",
        "Dense retrieval ranked the expected document first for all 10 questions.",
        "These observations apply to this controlled five-document, ten-question evaluation set.",
    ]
)

output_path.write_text("\n".join(lines), encoding="utf-8")

print(f"Manual evaluation evidence created: {output_path}")
