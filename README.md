# vhns' repository for work on ml (currently PKLot)

Currently this only includes some files from the work I'm doing on the
PKLot dataset as part of the research being done under PUCPR as a PIBIC
student.

There's not much to be said other than that you can run the code upon
making sure you have the dataset and installing the requirements by
doing:

`wget http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz`

`tar xpf PKLot.tar.gz`

`$python3 -m venv .venv`

`$source .venv/bin/activate`

`pip install -r requirements.txt`

`$python3 -i main.py`

The `-i` is recommended so that you can play with the model once it's
been generated. I also recommend fiddling with the layers as you see
fit.

There's much to be done, and is described in the [TODO](./TODO.md) file.
