#!/bin/bash
cd /mnt/data1/work/research-vlm-based-document-parsing

# Wait for papers validation to finish
while ps -p 1472237 --no-headers > /dev/null 2>&1; do
  sleep 120
done

# Generate summary
python3 -c "
import json

print()
print('=' * 70)
print('  PAPERS GT VALIDATION COMPLETE')
print('=' * 70)

r = json.load(open('datasets/papers/validation_report.json'))
print(f'  Judge Model: {r[\"judge_model\"]}')
print(f'  Total Sampled: {r[\"total_sampled_pages\"]}p')
print(f'  Overall Avg Score: {r[\"overall_avg_score\"]}/5')
print(f'  Overall Pass Rate: {r[\"overall_pass_rate\"]*100:.1f}%')
print(f'  Total Failed: {r[\"total_failed_pages\"]}')
print(f'  Total Time: {r[\"total_time_seconds\"]/60:.0f}min')
print()
print(f'{\"Document\":<15} {\"Sampled\":>8} {\"Avg\":>6} {\"Min\":>5} {\"Max\":>5} {\"Pass%\":>7} {\"Failed\":>8}')
print(f'{\"-\"*15} {\"-\"*8} {\"-\"*6} {\"-\"*5} {\"-\"*5} {\"-\"*7} {\"-\"*8}')

for d in r['documents']:
    failed = str(len(d['failed_pages'])) if d['failed_pages'] else '-'
    print(f'{d[\"doc_id\"]:<15} {d[\"sampled_pages\"]:>8} {d[\"avg_score\"]:>6.1f} {d[\"min_score\"]:>5} {d[\"max_score\"]:>5} {d[\"acceptable_ratio\"]*100:>6.0f}% {failed:>8}')

print(f'{\"-\"*15} {\"-\"*8} {\"-\"*6} {\"-\"*5} {\"-\"*5} {\"-\"*7} {\"-\"*8}')
print(f'{\"TOTAL\":<15} {r[\"total_sampled_pages\"]:>8} {r[\"overall_avg_score\"]:>6.1f} {\"\":>5} {\"\":>5} {r[\"overall_pass_rate\"]*100:>6.0f}% {r[\"total_failed_pages\"]:>8}')

# Error type analysis
error_types = {'think_tag': 0, 'truncation': 0, 'table_broken': 0, 'ocr_error': 0, 'other': 0}
for doc in r['documents']:
    for p in doc['page_results']:
        if not p['is_acceptable'] and p.get('errors'):
            for e in p['errors']:
                el = e.lower()
                if 'think' in el or 'chain-of-thought' in el or 'reasoning' in el or 'meta-commentary' in el:
                    error_types['think_tag'] += 1
                elif 'truncat' in el or 'abrupt' in el or 'incomplete' in el:
                    error_types['truncation'] += 1
                elif 'table' in el:
                    error_types['table_broken'] += 1
                elif 'ocr' in el or 'typo' in el or 'transcription' in el:
                    error_types['ocr_error'] += 1
                else:
                    error_types['other'] += 1

print()
print('=== Error Type Breakdown (FAIL pages only) ===')
for k, v in sorted(error_types.items(), key=lambda x: -x[1]):
    if v > 0:
        print(f'  {k}: {v}')

print()
print('=' * 70)
print('  Report: datasets/papers/validation_report.json')
print('=' * 70)
"
