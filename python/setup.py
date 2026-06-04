"""Marks the distribution as platform-specific so the wheel carries the right
tag (e.g. cp3-none-manylinux...) — the package bundles prebuilt native
libraries in quantajump/_libs/ rather than a pure-Python payload. Metadata
lives in pyproject.toml; this file only overrides the wheel platform tag.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # forces a platform (non-purelib) wheel tag
        return True


# The bundled libraries are loaded via ctypes, not linked against a specific
# CPython ABI, so the wheel is platform-specific but ABI-agnostic: tag it
# py3-none-<platform> rather than the building interpreter's cp3XX tag.
try:
    try:
        from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            _python, _abi, plat = super().get_tag()
            return "py3", "none", plat

    cmdclass = {"bdist_wheel": bdist_wheel}
except ImportError:  # pragma: no cover
    cmdclass = {}

setup(distclass=BinaryDistribution, cmdclass=cmdclass)
