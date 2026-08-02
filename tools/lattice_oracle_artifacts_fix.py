from pathlib import Path

path = Path("lattice/tests/test_integration.py")
text = path.read_text(encoding="utf-8")
start = text.index("with open(oracle_path, 'w', encoding='utf-8') as oracle:")
end = text.index("print('REPLAY_ORACLE_WRITTEN v2", start)
replacement = r'''with open(oracle_path, 'w', encoding='utf-8') as oracle:
    for forced in range(4, 12):
        print(
            f'STEP\\tv2\\t{forced}\\t2\\t4\\t0\\t3\\t1.25\\t0.5\\t0.8\\t1.9',
            '\\t1\\t-2\\t3\\t-4\\t2,4,3,1,5,6,7,8',
            sep='',
            file=oracle,
        )
    print('SUMMARY\\tv2\\t8\\t8\\tseparate_replay_pass\\tprivate_file', file=oracle)
'''
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
