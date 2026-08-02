import os
import subprocess
import urllib.request
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def main():
    base_dir = Path(os.path.abspath(__file__)).parent.parent
    external_dir = base_dir / "external"
    checkpoints_dir = base_dir / "checkpoints"
    
    external_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    
    # 1. Clone Repositories
    repos = {
        "CRM": "https://github.com/thu-ml/CRM.git",
        "Unique3D": "https://github.com/AiuniAI/Unique3D.git",
        "SAMPart3D": "https://github.com/Pointcept/SAMPart3D.git",
    }
    
    for name, url in repos.items():
        repo_path = external_dir / name
        if not (repo_path / ".git").exists():
            print(f"Cloning {name}...")
            run_cmd(f"git clone {url} {repo_path.name}", cwd=str(external_dir))
        else:
            print(f"{name} already cloned.")

    # 2. Download Checkpoints
    sam2_dir = checkpoints_dir / "sam2"
    sam2_dir.mkdir(exist_ok=True)
    sam2_ckpt = sam2_dir / "sam2_hiera_large.pt"
    if not sam2_ckpt.exists():
        print("Downloading SAM 2 checkpoint...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt",
            str(sam2_ckpt)
        )
    else:
        print("SAM 2 checkpoint already exists.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface-hub for model downloads...")
        run_cmd("python -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    crm_dir = checkpoints_dir / "crm"
    print("Downloading CRM models...")
    snapshot_download(repo_id="Zhengyi/CRM", local_dir=str(crm_dir))

    unique3d_dir = checkpoints_dir / "unique3d"
    print("Downloading Unique3D models...")
    snapshot_download(repo_id="Wuvin/Unique3D", local_dir=str(unique3d_dir))
    
    sampart3d_dir = checkpoints_dir / "sampart3d"
    print("Downloading SAMPart3D models...")
    snapshot_download(repo_id="yhyang-myron/SAMPart3D", local_dir=str(sampart3d_dir))
    
    print("All downloads complete!")

if __name__ == "__main__":
    main()
