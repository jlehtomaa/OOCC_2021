# OOCC_2021

This repository accompanies the open-access publication *Solar Geoengineering Governance: A Dynamic Framework of Farsighted Coalition Formation (Heyen & Lehtomaa, Oxford Open Climate Change, 2021)*.


Running the code verifies all equilibria and replicates all numerical results discussed in the paper.
The paper can be accessed on the [journal's website](https://academic.oup.com/oocc/issue/1/1).
To cite this work, please use:

```
@article{10.1093/oxfclm/kgab010,
    author = {Heyen, Daniel and Lehtomaa, Jere},
    title = "{Solar Geoengineering Governance: A Dynamic Framework of Farsighted Coalition Formation}",
    journal = {Oxford Open Climate Change},
    year = {2021},
    month = {09},
    issn = {2634-4068},
    doi = {10.1093/oxfclm/kgab010},
    url = {https://doi.org/10.1093/oxfclm/kgab010},
    note = {kgab010},
    eprint = {https://academic.oup.com/oocc/advance-article-pdf/doi/10.1093/oxfclm/kgab010/40392174/kgab010.pdf},
}
```

### Requirements
Running the code relies on minimum dependencies: only `numpy` and `pandas` (and `pytest` for testing) are required.

### Running the code
To replicate all results in the paper, simply run ```python main.py```.
All results will appear in the ```results``` folder.
For testing different player strategies, directly modify the tables in the ```strategy_tables``` folder.
To try out different model parameterizations (discount rates, base temperatures, marginal damages, protocols, etc.), 
modify the ```base_config``` and ```experiment_configs``` variables inside ```main.py```.