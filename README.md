# OOCC_2021

This is an accompanying code repository for the open-access publication *Solar Geoengineering Governance: A Dynamic Framework of Farsighted Coalition Formation (Heyen & Lehtomaa, 2021)*.
Running the code replicates and verifies all equilibria and numerical results discussed in the paper.
The paper can be accessed on the [journal's website](https://academic.oup.com/oocc/issue/1/1).
To cite this work, please use:

```
@article{heyen2021solar,
    title = {Solar Geoengineering Governance: A Dynamic Framework of Farsighted Coalition Formation},
    journal = {Oxford Open Climate Change},
    volume = {1},
    issn = {xxx},
    url = {xxx},
    doi = {xxx},
    urldate = {xxx},
    author = {Heyen, Daniel and Lehtomaa, Jere},
    month = {sep},
    year = {2021},
    keywords = {xxx},
    pages = {xxx},
}
```

### Requirements
Running the code relies on minimum dependencies: only `numpy`, `pandas`, (and `pytest` for testing) are required.

### Running the code
To replicate all results in the paper, simply run ```python main.py```.
All results will appear in the ```results``` folder.
For testing different player strategies, directly modify the tables in the ```strategy_tables``` folder.
To try out different model parameterizations (discount rates, base temperatures, marginal damages, protocols, etc.), 
modify the ```base_config``` and ```experiment_configs``` variables inside ```main.py```.