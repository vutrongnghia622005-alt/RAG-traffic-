from pathlib import Path
import argparse
import json
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'scripts' else Path.cwd()
if (PROJECT_ROOT / 'src').exists() is False:
    # Fallback when the script is copied into scripts/ and run from project root.
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / 'src').exists():
        PROJECT_ROOT = candidate

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rag_pipeline import AdvancedRAGPipeline


QUICK_CASES = [
    {
        'id': 'demo01',
        'question': 'Điểm của giấy phép lái xe được dùng để làm gì?',
        'expected_status': 'answered',
        'expected_article': '58',
        'purpose': 'Câu supported cơ bản + citation',
    },
    {
        'id': 'demo02',
        'question': 'Theo Điều 87 khoản 2, Bộ Công an có trách nhiệm gì trong quản lý nhà nước về trật tự, an toàn giao thông đường bộ?',
        'expected_status': 'answered',
        'expected_article': '87',
        'purpose': 'Hard routing/filter theo Điều + Khoản',
    },
    {
        'id': 'demo03',
        'question': 'Giá xăng RON95 hiện nay là bao nhiêu một lít?',
        'expected_status': 'abstain',
        'expected_article': None,
        'purpose': 'OOD rõ ràng, phải từ chối trước generation',
    },
    {
        'id': 'demo04',
        'question': 'Theo Điều 58, lệ phí cấp giấy phép lái xe là bao nhiêu tiền?',
        'expected_status': 'abstain',
        'expected_article': None,
        'purpose': 'Hard-negative có số Điều đúng nhưng nội dung không có trong luật',
    },
]

FULL_CASES = QUICK_CASES + [
    {
        'id': 'demo05',
        'question': 'Chỉ huy giao thông đường bộ là hoạt động gì?',
        'expected_status': 'answered',
        'expected_article': None,
        'purpose': 'Câu định nghĩa/chức năng',
    },
    {
        'id': 'demo06',
        'question': 'Theo Điều 75, Trung tâm chỉ huy giao thông đường bộ có nhiệm vụ gì?',
        'expected_status': 'answered',
        'expected_article': '75',
        'purpose': 'Câu supported có explicit article filter',
    },
    {
        'id': 'demo07',
        'question': 'Mức phạt tiền đối với ô tô vượt đèn đỏ là bao nhiêu?',
        'expected_status': 'abstain',
        'expected_article': None,
        'purpose': 'Near-domain unsupported',
    },
    {
        'id': 'demo08',
        'question': 'Tra cứu phạt nguội phương tiện trực tuyến ở website nào?',
        'expected_status': 'abstain',
        'expected_article': None,
        'purpose': 'Thông tin dịch vụ bên ngoài tài liệu',
    },
]


def _fmt_float(value, digits=4):
    if value is None:
        return '-'
    try:
        return f'{float(value):.{digits}f}'
    except Exception:
        return str(value)


def _retrieval_seconds(timing):
    value = (timing or {}).get('retrieval')
    if isinstance(value, dict):
        return value.get('total')
    return value


def evaluate_case(case, result):
    actual_status = result.get('status')
    expected_status = case['expected_status']
    status_ok = actual_status == expected_status

    guardrail = result.get('guardrail') or {}
    citation = result.get('citation_validation')
    sources = result.get('sources') or []
    timing = result.get('timing') or {}

    citation_ok = None
    if expected_status == 'answered':
        citation_ok = bool(citation and citation.get('valid'))

    article_ok = None
    expected_article = case.get('expected_article')
    if expected_status == 'answered' and expected_article is not None:
        actual_articles = {str(s.get('article_number')) for s in sources}
        article_ok = str(expected_article) in actual_articles

    bypass_ok = None
    if expected_status == 'abstain':
        generation_seconds = float(timing.get('generation', 0.0) or 0.0)
        bypass_ok = generation_seconds == 0.0

    checks = [status_ok]
    if citation_ok is not None:
        checks.append(citation_ok)
    if article_ok is not None:
        checks.append(article_ok)
    if bypass_ok is not None:
        checks.append(bypass_ok)

    overall_pass = all(checks)

    return {
        'id': case['id'],
        'purpose': case['purpose'],
        'question': case['question'],
        'expected_status': expected_status,
        'actual_status': actual_status,
        'pass': overall_pass,
        'status_ok': status_ok,
        'guardrail_supported': guardrail.get('supported'),
        'guardrail_score': guardrail.get('top1_score'),
        'guardrail_threshold': guardrail.get('threshold'),
        'citation_ok': citation_ok,
        'cited_source_ids': (citation or {}).get('cited_source_ids') if citation else [],
        'expected_article': expected_article,
        'article_ok': article_ok,
        'bypass_ok': bypass_ok,
        'answer': result.get('answer'),
        'generation_error': result.get('generation_error'),
        'sources': [
            {
                'source_id': s.get('source_id'),
                'article_number': s.get('article_number'),
                'clause': s.get('clause'),
                'page_start': s.get('page_start'),
                'page_end': s.get('page_end'),
                'rerank_score': s.get('rerank_score'),
            }
            for s in sources
        ],
        'timing': {
            'retrieval_seconds': _retrieval_seconds(timing),
            'reranker_seconds': timing.get('reranker'),
            'generation_seconds': timing.get('generation'),
            'total_seconds': timing.get('total'),
        },
    }


def print_case_report(record, index, total):
    print('\n' + '=' * 96)
    print(f'RAG DEMO TEST [{index}/{total}] {record["id"]}')
    print('=' * 96)
    print(f'Mục đích  : {record["purpose"]}')
    print(f'Câu hỏi   : {record["question"]}')
    print(f'Expected  : {record["expected_status"]}')
    print(f'Actual    : {record["actual_status"]}')
    print(f'PASS      : {record["pass"]}')
    print('-' * 96)
    print(
        'Guardrail : '
        f'supported={record["guardrail_supported"]} | '
        f'score={_fmt_float(record["guardrail_score"])} | '
        f'threshold={_fmt_float(record["guardrail_threshold"])}'
    )
    if record['citation_ok'] is not None:
        print(f'Citation  : valid={record["citation_ok"]} | cited={record["cited_source_ids"]}')
    if record['article_ok'] is not None:
        print(f'Article   : expected={record["expected_article"]} | present={record["article_ok"]}')
    if record['bypass_ok'] is not None:
        print(f'Bypass LLM: {record["bypass_ok"]}')
    if record.get('generation_error'):
        print(f'Gen error : {record["generation_error"]}')

    print('-' * 96)
    if record.get('answer'):
        print('ANSWER')
        print(record['answer'])
    else:
        print('ANSWER: None')

    if record['sources']:
        print('-' * 96)
        print('TOP SOURCES')
        for s in record['sources'][:5]:
            print(
                f'[{s.get("source_id")}] Điều {s.get("article_number")} | '
                f'Khoản {s.get("clause")} | Trang {s.get("page_start")}-{s.get("page_end")} | '
                f'Score {_fmt_float(s.get("rerank_score"))}'
            )

    t = record['timing']
    print('-' * 96)
    print(
        'TIMING    : '
        f'retrieval={_fmt_float(t.get("retrieval_seconds"))}s | '
        f'rerank={_fmt_float(t.get("reranker_seconds"))}s | '
        f'generation={_fmt_float(t.get("generation_seconds"))}s | '
        f'total={_fmt_float(t.get("total_seconds"))}s'
    )


def summarize(records):
    total = len(records)
    passed = sum(1 for r in records if r['pass'])
    status_correct = sum(1 for r in records if r['status_ok'])
    answered = [r for r in records if r['expected_status'] == 'answered']
    abstain = [r for r in records if r['expected_status'] == 'abstain']
    citation_eval = [r for r in answered if r['citation_ok'] is not None]
    bypass_eval = [r for r in abstain if r['bypass_ok'] is not None]
    gen_errors = [r for r in records if r.get('generation_error') or r.get('actual_status') in {'generation_error', 'generator_unavailable'}]
    totals = [float(r['timing']['total_seconds']) for r in records if r['timing'].get('total_seconds') is not None]

    return {
        'n_total': total,
        'n_passed': passed,
        'pass_rate': passed / total if total else None,
        'status_accuracy': status_correct / total if total else None,
        'citation_validity_rate': (
            sum(1 for r in citation_eval if r['citation_ok']) / len(citation_eval)
            if citation_eval else None
        ),
        'unsupported_bypass_rate': (
            sum(1 for r in bypass_eval if r['bypass_ok']) / len(bypass_eval)
            if bypass_eval else None
        ),
        'generation_error_rate': len(gen_errors) / total if total else None,
        'mean_total_seconds': statistics.mean(totals) if totals else None,
        'failed_ids': [r['id'] for r in records if not r['pass']],
    }


def print_summary(summary):
    print('\n' + '=' * 96)
    print('RAG DEMO SUMMARY')
    print('=' * 96)
    print(f'Total test cases             : {summary["n_total"]}')
    print(f'Passed                       : {summary["n_passed"]}')
    print(f'Overall pass rate            : {_fmt_float(summary["pass_rate"])}')
    print(f'Status accuracy              : {_fmt_float(summary["status_accuracy"])}')
    print(f'Citation validity rate       : {_fmt_float(summary["citation_validity_rate"])}')
    print(f'Unsupported generation bypass: {_fmt_float(summary["unsupported_bypass_rate"])}')
    print(f'Generation error rate        : {_fmt_float(summary["generation_error_rate"])}')
    print(f'Mean total latency           : {_fmt_float(summary["mean_total_seconds"])} s')
    print(f'Failed IDs                   : {summary["failed_ids"]}')
    print('=' * 96)


def parse_args():
    parser = argparse.ArgumentParser(description='Demo/test Advanced RAG for teacher presentation.')
    parser.add_argument('--suite', choices=['quick', 'full'], default='quick', help='quick=4 cases, full=8 cases')
    parser.add_argument('--interactive', action='store_true', help='Run interactive Q&A after the test suite')
    parser.add_argument('--no-generation', action='store_true', help='Disable Qwen generation for retrieval-only demo')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/demo_rag_test.json'),
        help='JSON output path relative to project root',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = QUICK_CASES if args.suite == 'quick' else FULL_CASES

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print('=' * 96)
    print('ADVANCED RAG - DEMO TEST SUITE')
    print('=' * 96)
    print(f'Suite       : {args.suite} ({len(cases)} cases)')
    print(f'Generation  : {not args.no_generation}')
    print(f'Output      : {output_path}')
    print('Note        : Keep llama-server running when generation is enabled.')

    pipeline = AdvancedRAGPipeline(enable_generation=not args.no_generation)
    records = []

    for idx, case in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            result = pipeline.ask(case['question'])
            record = evaluate_case(case, result)
        except Exception as exc:
            record = {
                'id': case['id'],
                'purpose': case['purpose'],
                'question': case['question'],
                'expected_status': case['expected_status'],
                'actual_status': 'error',
                'pass': False,
                'status_ok': False,
                'guardrail_supported': None,
                'guardrail_score': None,
                'guardrail_threshold': None,
                'citation_ok': None,
                'cited_source_ids': [],
                'expected_article': case.get('expected_article'),
                'article_ok': None,
                'bypass_ok': None,
                'answer': None,
                'generation_error': f'{type(exc).__name__}: {exc}',
                'sources': [],
                'timing': {
                    'retrieval_seconds': None,
                    'reranker_seconds': None,
                    'generation_seconds': None,
                    'total_seconds': time.perf_counter() - started,
                },
            }
        records.append(record)
        print_case_report(record, idx, len(cases))

    summary = summarize(records)
    print_summary(summary)

    payload = {
        'suite': args.suite,
        'generation_enabled': not args.no_generation,
        'summary': summary,
        'cases': records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved report: {output_path}')

    if args.interactive:
        print('\nInteractive mode. Type exit to quit.')
        while True:
            q = input('\nCâu hỏi > ').strip()
            if q.lower() in {'exit', 'quit', 'q'}:
                break
            if not q:
                continue
            result = pipeline.ask(q)
            print(f'\nStatus: {result.get("status")}')
            print(f'Answer: {result.get("answer")}')
            citation = result.get('citation_validation')
            if citation:
                print(f'Citation valid: {citation.get("valid")} | Cited: {citation.get("cited_source_ids")}')
            guardrail = result.get('guardrail') or {}
            print(
                'Guardrail: '
                f'supported={guardrail.get("supported")} | '
                f'score={_fmt_float(guardrail.get("top1_score"))} | '
                f'threshold={_fmt_float(guardrail.get("threshold"))}'
            )


if __name__ == '__main__':
    main()
