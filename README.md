# What does automatic differentiation compute for neural networks? (ICLR 2024 spotlight)

[![openreview](https://img.shields.io/badge/OpenReview%20-forum-b31b1b.svg)](https://openreview.net/forum?id=8vKknbgXxf)

[Sejun Park*](https://sites.google.com/site/sejunparksite/), [Sanghyuk Chun*](https://sanghyukchun.github.io/home/), [Wonyeol Lee](https://wonyeol.github.io/)

## How to run?

```
$ python main.py --model vgg11 --log_dir logs/cnns/vgg11
```

You can check if your training code passed the correctness test or not by:

```
$ grep False ./logs/cnns/vgg11/ad_log.txt
```

The full scripts for each network can be found in `scripts/`

If you have any `False` in `ad_log.txt`, it means that the AD checker has been failed.

## Reference codes

- https://github.com/Fangyh09/pytorch-receptive-field/blob/master/torch_receptive_field/receptive_field.py
- https://github.com/kuangliu/pytorch-cifar

## How to cite?

```
@inproceedings{park2024autodiff_correctness,
    title={What does automatic differentiation compute for neural networks?},
    author={Sejun Park and Sanghyuk Chun and Wonyeol Lee},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=8vKknbgXxf}
}
```
