# asm2vec

This is an unofficial implementation of the `asm2vec` model as a standalone python package. The details of the model can be found in the original paper: [(sp'19) Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search against Code Obfuscation and Compiler Optimization](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a038/19skfc3ZfKo)

## Requirements

This implementation is written in python 3.7 and it's recommended to use python 3.7+ as well. The only dependency of this package is `numpy` which can be installed as follows:

```shell
python3 -m pip install numpy
```

## How to use

### Import

To install the package, execute the following commands:

```shell
git clone https://github.com/lancern/asm2vec.git
```

Add the following line to the `.bashrc` file to add `asm2vec` to your python interpreter's search path for external packages:

```shell
export PYTHONPATH="path/to/asm2vec:$PYTHONPATH"
```

Replace `path/to/asm2vec` with the directory you clone `asm2vec` into. Then execute the following commands to update `PYTHONPATH`:

```shell
source ~/.bashrc
```

You can also add the following code snippets to your python source code referring `asm2vec` to guide python interpreter finding the package successfully:

```python
import sys
sys.path.append('path/to/asm2vec')
```

In your python code, use the following `import` statement to import this package:

```python
import asm2vec.<module-name>
```

### Define CFGs And Training

You have 2 approaches to define the binary program that will be sent to the `asm2vec` model. The first approach is to build the CFG manually, as shown below:

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

And then you can train a model with the following code:

```python
from asm2vec.model import Asm2Vec

model = Asm2Vec(d=200)
train_repo = model.make_function_repo([f1, f2, f3])
model.train(train_repo)
```

The second approach is using the `parse` module provided by `asm2vec` to build CFGs automatically from an assembly code source file:

```python
from asm2vec.parse import parse_fp

with open('source.asm', 'r') as fp:
    funcs = parse_fp(fp)
```

And then you can train a model with the following code:

```python
from asm2vec.model import Asm2Vec

model = Asm2Vec(d=200)
train_repo = model.make_function_repo(funcs)
model.train(train_repo)
```

### Estimation

You can use the `asm2vec.model.Asm2Vec.to_vec` method to convert a function into its vector representation.

### Serialization

The implementation support serialization on many of its internal data structures so that you can serialize the internal state of a trained model into disk for future use.

You can serialize two data structures to primitive data: the function repository and the model memento.

> To be finished.

## Hyper Parameters

The constructor of `asm2vec.model.Asm2Vec` class accepts some keyword arguments as hyper parameters of the model. The following table lists all the hyper parameters available:

| Parameter Name          | Type    | Meaning                                                                                                | Default Value |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------------------ | ------------- |
| `d`                     | `int`   | The dimention of the vectors for tokens.                                                               | `200`         |
| `initial_alpha`         | `float` | The initial learning rate.                                                                             | `0.05`        |
| `alpha_update_interval` | `int`   | How many tokens can be processed before changing the learning rate?                                    | `10000`       |
| `rnd_walks`             | `int`   | How many random walks to perform to sequentialize a function?                                          | `3`           |
| `neg_samples`           | `int`   | How many samples to take during negative sampling?                                                     | `25`          |
| `iteration`             | `int`   | How many iterations to perform? (This parameter is reserved for future use and is not implemented now) | `1`           |
| `jobs`                  | `int`   | How many tasks to execute concurrently during training?                                                | `4`           |

## Notes

For simplicity, the Selective Callee Expansion is not implemented in this early implementation. You have to do it manually before sending CFG into `asm2vec` .
