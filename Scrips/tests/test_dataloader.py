from pathlib import Path

def test_data_dirs_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "Data").exists()

if __name__ == "__main__":
    test_data_dirs_exist()
    print("test_dataloader OK")
