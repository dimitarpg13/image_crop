# Installation instructions and Running unit tests 

## On MacOS

* make sure you have at least poetry 1.8.3 installed
```
➜  image_crop git:(main) poetry --version
Configuration file exists at /Users/[your_username]/Library/Preferences/pypoetry, reusing this directory.

Consider moving TOML configuration files to /Users/[your_username]/Library/Application Support/pypoetry, as support for the legacy directory will be removed in an upcoming release.
Poetry (version 1.8.3)
```

* configure python 3.10.x to be your default python version via pyenv
```
➜  image_crop git:(main) pyenv local 3.10.10

➜  image_crop git:(main) pyenv versions
  system
  3.7.10
  3.7.12
* 3.10.10 (set by /Users/[your_username]/.pyenv/version)
  3.12.4
```

* set this version of python to be the default poetry environment
```
poetry env use 3.10.10
```

* fetch the submodules of this repo
```
➜  image_crop git:(main) ✗ git submodule init
Submodule 'image_crop/dichotomous_image_segmentation' (https://github.com/dimitarpg13/dichotomous_image_segmentation.git) registered for path 'image_crop/dichotomous_image_segmentation'

➜  image_crop git:(main) ✗ git submodule update --recursive --remote
Cloning into '/Users/[your_username]/image_crop/dichotomous_image_segmentation'...
Submodule path 'image_crop/dichotomous_image_segmentation': checked out '13d14299d03890733f972419a839b9d3df087bc5
```

* install the whole repo
```
➜  image_crop git:(main) ✗ poetry install
Configuration file exists at /Users/[your_user_name]/Library/Preferences/pypoetry, reusing this directory.

Consider moving TOML configuration files to /Users/[your_user_name]/Library/Application Support/pypoetry, as support for the legacy directory will be removed in an upcoming release.
Installing dependencies from lock file

Package operations: 137 installs, 1 update, 0 removals

  - Installing attrs (23.2.0)
  - Installing rpds-py (0.18.1)
  - Installing referencing (0.35.1)
...
```

* run the image_crop unit tests
```
➜  image_crop git:(main) ✗ poetry run python image_crop/tests/test_image_crop_manager.py
Configuration file exists at /Users/[your_username]/Library/Preferences/pypoetry, reusing this directory.

Consider moving TOML configuration files to /Users/[your_username]/Library/Application Support/pypoetry, as support for the legacy directory will be removed in an upcoming release.
/Users/[your_username]/image_crop/.venv/lib/python3.10/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='mean' instead.
  warnings.warn(warning.format(ret))
/Users/[your_username]/image_crop/.venv/lib/python3.10/site-packages/torch/nn/functional.py:3782: UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
.
----------------------------------------------------------------------
Ran 2 tests in 38.894s

OK

``` 
