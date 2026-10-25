import json
import subprocess
import tarfile
from scripts.package_release import package


def test_release_excludes_data_credentials_and_outputs(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    for name in ['README.md', 'code.py', '.env', '.env.example', 'data/case.json',
                 'outputs/result.md', 'weights/model.bin', 'config/local.yaml',
                 'secrets.json', 'synthea_extensions/modules/example.json', 'synthea_extensions/NOTICE']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic')
    output = tmp_path / 'release.tar.gz'
    package(output, root=tmp_path)
    with tarfile.open(output) as archive:
        names = set(archive.getnames())
    assert 'ClinVerse/.env.example' in names
    assert 'ClinVerse/code.py' in names
    assert 'ClinVerse/synthea_extensions/NOTICE' in names
    assert 'ClinVerse/synthea_extensions/modules/example.json' in names
    assert not any(n in names for n in ['ClinVerse/.env', 'ClinVerse/data/case.json',
                   'ClinVerse/outputs/result.md', 'ClinVerse/config/local.yaml', 'ClinVerse/secrets.json'])
