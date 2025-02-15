import pathlib
import shutil

def create_directory(path: str) -> None:
    """ディレクトリを作成する関数

    Args:
        path(str): 作成するディレクトリのパス

    Example:
        >>> from VitLib import create_directory
        >>> create_directory('example_dir')
    """
    p = pathlib.Path(path)
    if not p.exists():
        p.mkdir(parents=True)

def delete_directory(path: str) -> None:
    """ディレクトリを削除する関数

    Args:
        path(str): 削除するディレクトリのパス

    Example:
        >>> from VitLib import delete_directory
        >>> delete_directory('example_dir')
    """
    if file_exists(path):
        shutil.rmtree(pathlib.Path(path))

def get_file_paths(path: str) -> list[str]:
    """フォルダ内のファイルとフォルダのパスをすべて取得する関数

    Args:
        path(str): フォルダのパス

    Returns:
        list[str]: フォルダ内のすべてのファイルとフォルダのパスリスト

    Example:
        >>> from VitLib import get_file_paths
        >>> get_file_paths('example_dir')
    """
    folder_path = pathlib.Path(path)
    file_paths = list(folder_path.glob('*'))
    return [str(path).replace('\\', '/') for path in file_paths]

def get_file_names(path: str) -> list[str]:
    """フォルダ内のファイル名をすべて取得する関数

    Args:
        path(str): フォルダのパス

    Returns:
        list[str]: フォルダ内のすべてのファイル名のリスト

    Example:
        >>> from VitLib import get_file_names
        >>> get_file_names('example_dir')
    """
    folder_path = pathlib.Path(path)
    file_paths = list(folder_path.glob('*'))
    return [str(path.name) for path in file_paths]

def get_file_stems(path: str) -> list[str]:
    """フォルダ内のファイル名（拡張子なし）をすべて取得する関数

    Args:
        path(str): フォルダのパス

    Returns:
        list[str]: フォルダ内のすべてのファイル名（拡張子なし）のリスト

    Example:
        >>> from VitLib import get_file_stems
        >>> get_file_stems('example_dir')
    """
    folder_path = pathlib.Path(path)
    file_paths = list(folder_path.glob('*'))
    return [str(path.stem) for path in file_paths]

def file_exists(path: str) -> bool:
    """ファイルが存在するかどうかを返す関数

    Args:
        path(str): ファイルのパス

    Returns:
        bool: ファイルが存在する場合はTrue、そうでない場合はFalse

    Example:
        >>> from VitLib import file_exists
        >>> file_exists('example_dir/example_file.txt')
    """
    return pathlib.Path(path).exists()

def delete_file(path: str) -> None:
    """ファイルを削除する関数

    Args:
        path(str): 削除するファイルのパス

    Example:
        >>> from VitLib import delete_file
        >>> delete_file('example_dir/example_file.txt')
    """
    pathlib.Path(path).unlink()
