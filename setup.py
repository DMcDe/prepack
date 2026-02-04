from setuptools import setup

setup(name="prepack",
      version="0.3",
      description="Efficient offline microbatch packer for AI training (WIP)",
      url="https://github.com/DMcDe/prepack",
      author="David McDermott",
      packages=["prepack", "prepack.containers", "prepack.packing_functions"])