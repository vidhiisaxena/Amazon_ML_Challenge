import pandas as pd

# Ground truth analysis
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype=str)
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
print(f'Total S1 entities in ground truth: {len(gt)}')
singletons = (gt['matched_entity_ids'] == '').sum()
print(f'With matches (non-singleton): {len(gt) - singletons}')
print(f'Without matches (singleton): {singletons}')

# Count avg matches
match_counts = gt['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
non_zero = match_counts[match_counts > 0]
print(f'Avg matches (non-singleton): {non_zero.mean():.2f}')
print(f'Max matches: {match_counts.max()}')

# Country distribution
for src in ['dataset/train/train_source1.tsv', 'dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    df = pd.read_csv(src, sep='\t', usecols=['country'])
    print(f'\n{src} country distribution:')
    print(df['country'].value_counts().to_string())

# Check for nulls in key columns
print('\n--- Null counts ---')
for src in ['dataset/train/train_source1.tsv', 'dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    df = pd.read_csv(src, sep='\t')
    print(f'\n{src}:')
    print(df.isnull().sum().to_string())
