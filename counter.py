from src.data import load_json_records

records = load_json_records("data/raw/dataset.json")


print(f"Total records: {len(records)}")
print(f"Total vulnerable records: {len([r for r in records if r['target'] == 1])}")
print(f"Total non-vulnerable records: {len([r for r in records if r['target'] == 0])}")
