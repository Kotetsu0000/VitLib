import pathlib
import shutil

def create_directory(path: str) -> None:
    """ディレクトリを作成する関数

    Args:
        path: 作成するディレクトリのパス
    """
    p = pathlib.Path(path)
    if not p.exists():
        p.mkdir(parents=True)

def delete_directory(path: str) -> None:
    """ディレクトリを削除する関数

    Args:
        path: 削除するディレクトリのパス
    """
    if file_exists(path):
        shutil.rmtree(pathlib.Path(path))

def get_file_paths(path: str) -> list[str]:
    """フォルダ内のファイルとフォルダのパスをすべて取得する関数

    Args:
        path: フォルダのパス

    Returns:
        list[str]: フォルダ内のすべてのファイルとフォルダのパスリスト
    """
    folder_path = pathlib.Path(path)
    file_paths = list(folder_path.glob('*'))
    return [str(path).replace('\\', '/') for path in file_paths]

def get_file_names(path: str) -> list[str]:
    """フォルダ内のファイル名をすべて取得する関数

    Args:
        path: フォルダのパス

    Returns:
        list[str]: フォルダ内のすべてのファイル名のリスト
    """
    folder_path = pathlib.Path(path)
    file_paths = list(folder_path.glob('*'))
    return [str(path.name) for path in file_paths]

def get_file_stems(path: str) -> list[str]:
    """フォルダ内のファイル名（拡張子なし）をすべて取得する関数

    Args:
        path: フォルダのパス

    Returns:
        list[str]: フォルダ内のすべてのファイル名（拡張子なし）のリスト
    """
    folder_path = pathlib.Path(path)
    file_paths = list(folder_path.glob('*'))
    return [str(path.stem) for path in file_paths]

def file_exists(path: str) -> bool:
    """ファイルが存在するかどうかを返す関数

    Args:
        path: ファイルのパス

    Returns:
        bool: ファイルが存在する場合はTrue、そうでない場合はFalse
    """
    return pathlib.Path(path).exists()

def delete_file(path: str) -> None:
    """ファイルを削除する関数

    Args:
        path: 削除するファイルのパス
    """
    pathlib.Path(path).unlink()
