from pathlib import Path

def test_data_dirs_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "data").exists()

if __name__ == "__main__":
    test_data_dirs_exist()
    print("test_dataloader OK")
