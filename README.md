# asm2vec

This is an unofficial implementation of the `asm2vec` model as a standalone python package. The details of the model can be found in the original paper: [(sp'19) Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search against Code Obfuscation and Compiler Optimization](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a038/19skfc3ZfKo)

## Requirements

This implementation is written in python 3.7 and it's recommended to use python 3.7+ as well. The only dependency of this package is `numpy` which can be installed as follows:

```shell
python3 -m pip install numpy
```

## How to use

To install the package, execute the following commands:

```shell
git clone https://github.com/lancern/asm2vec
```

Add the following line to the `.bashrc` file to add `asm2vec` to your python interpreter's search path for external packages:

```shell
export PYTHONPATH="path/to/asm2vec:$PYTHONPATH"
```

Replacement `path/to/asm2vec` with the directory you clone `asm2vec` into. Then execute the following commands to update `PYTHONPATH`:

```shell
source ~/.bashrc
```

In your python code, use the following `import` statement to import this package:

```python
import asm2vec
```

In the early versions of this implementation, you have to build the control flow graph by your own:

```python
from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import parse_instruction

block1 = BasicBlock()
block1.add_instruction(parse_instruction('mov eax, ebx'))
block1.add_instruction(parse_instruction('jmp _loc'))

block2 = BasicBlock()
block2.add_instruction(parse_instruction('xor eax, eax'))
block2.add_instruction(parse_instruction('ret'))

block1.add_successor(block2)

block3 = BasicBlock()
block3.add_instruction(parse_instruction('sub eax, [ebp]'))

f1 = Function(block1, 'some_func')
f2 = Function(block3, 'another_func')

# block4 is ignore here for clarity
f3 = Function(block4, 'estimate_func')
```

You can create, train an `asm2vec` model and then estimate a test function by:

```python
from asm2vec.model import Asm2Vec

model = Asm2Vec(d=200, initial_alpha=0.005, rnd_walks=25)
model.train([f1, f2])
print(model.to_vec(f3))
```

And it's done.

## Hyper Parameters

The example below uses 3 hyper parameters of `asm2vec` model: `d`, `initial_alpha` and `rnd_walks`, which are set to `200`, `0.005`, `25` respectively. The following table lists all available hyper parameters:

| Parameter Name          | Type    | Meaning                                                                                               | Default Value |
| ----------------------- | ------- | ----------------------------------------------------------------------------------------------------- | ------------- |
| `d`                     | `int`   | The dimention of the vectors for tokens.                                                              | `200`         |
| `initial_alpha`         | `float` | The initial learning rate.                                                                            | `0.05`        |
| `alpha_update_interval` | `int`   | How many tokens can be processed before changing the learning rate?                                   | `10000`       |
| `rnd_walks`             | `int`   | How many random walks to perform to sequentialize a function?                                         | `3`           |
| `neg_samples`           | `int`   | How many samples to take during negative sampling?                                                    | `25`          |
| `iteration`             | `int`   | How many iteration to perform? (This parameter is reserved for future use and is not implemented now) | `1`           |
