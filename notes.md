# Entrypoints in Python Projects

The document imag denoising project has four entry points:

  1. denoising-extract-crops — one-time data prep. Point it at a folder of PDFs, get a folder of clean
  512×512 PNG crops out. Run once before anything else.
  2. denoising-train — training. Takes a dataset of clean crops, synthesizes JPEG noise on the fly,
  trains the model. Writes checkpoints and CSV logs.
  3. denoising-predict — visual QA. Loads a checkpoint, samples crops from your dataset, runs the model,
  and produces a demo.png grid showing noisy / denoised / clean side by side. Not for production use —
  just for eyeballing quality.
  4. denoising-infer — the actual inference interface. Takes any image off disk (arbitrary size), tiles
  it, denoises it, saves the result. This is what you'd use on a real document.

Notice that there are 4 underlying scrips that implement each:

src/denoising/train.py, etc etc.

We use Python's ENTRY POINTS which are declared in pyproject.toml

[project.scripts]
denoising-train = "denoising.train:main"
denoising-predict = "denoising.predict:main"
denoising-infer = "denoising.infer:main"
denoising-extract-crops = "denoising.data.extract_crops:main"


then, when you run pip install -e, a small executable script is generated and placed in the current env's bin. the desnoising-strain script goes simply like this


```
from fenoising.train import main
main()
```

this is standard for any published Python package

# Hydra: CLI overrides

main is decorated with @hydra.main, which means that Hydra intercepts sys.argv and applies them on top of the composed config before handing `cfg`to `main(cfg)`.

Any project using @hydra.main gets this for free — you write zero argument parsing code, no argparse, no click. Hydra owns the CLI layer entirely.

# Adding new models

the repo makes it super easy to experiment with new models. just define
a new model in src/denoising/models.

you can either entirely code your new model in pytorch (e.g. unet)
or you can import your model (swin2sr from hugging face).

