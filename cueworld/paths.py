"""パス解決。親リポジトリに置いたままでも、cueworld 単体でも同じコードで動く。"""
import os

# このパッケージのディレクトリ。URDF と objects/assets の基準。
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def project_root() -> str:
    """結果の出力先。

    consts/config.py がある親リポジトリの中では、そのリポジトリのルート。
    cueworld ディレクトリ自身がリポジトリのときは、このパッケージ直下。
    """
    parent = os.path.dirname(PACKAGE_DIR)
    if os.path.isfile(os.path.join(parent, "consts", "config.py")):
        return parent
    return PACKAGE_DIR


# デモの保存先。中身は project_root()。
ABS_PATH = project_root()
